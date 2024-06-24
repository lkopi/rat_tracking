import os

import luigi
import numpy as np

import foreground_segmentation
import automatic_annotation
import annotation2coco
import propagate_instances
import train_detectron2
import handannot2coco
import iterative_training
import behavior_detection_rule_based as behavior
import switch_up_behaviors
import utils.io
import run_gma
import position_statistics
import visualization
import confidence
import fix_sequences


class GlobalParams(luigi.Config):
    crop = luigi.ListParameter(default=[0, 0, 640, 420])
    n_processes = luigi.IntParameter(default=-1)
    chunksize = luigi.IntParameter(default=300)
    batch_size = luigi.IntParameter(default=4)  #6)  # VRAM // 1.2


class ParallelTask(luigi.Task):
    def __init__(self, *args, **kwargs):
        super(ParallelTask, self).__init__(*args, **kwargs)
        self.n_processes = GlobalParams().n_processes
        self.chunksize = GlobalParams().chunksize


class GPUTask(luigi.Task):
    def __init__(self, *args, **kwargs):
        super(GPUTask, self).__init__(*args, **kwargs)
        self.batch_size = GlobalParams().batch_size


class PrepareVideo(ParallelTask):
    video = luigi.Parameter()
    n_sequences = luigi.IntParameter(default=16)

    def __init__(self, *args, **kwargs):
        super(PrepareVideo, self).__init__(*args, **kwargs)
        self.out_root = os.path.join(os.path.splitext(self.video)[0])
        self.crop = GlobalParams().crop

    def run(self):
        foreground_segmentation.split_and_crop_video(video_in=self.video,
                                                     out_dir=self.out_root,
                                                     crop_bbox_xyxy=self.crop,
                                                     n_sequences=self.n_sequences,
                                                     n_processes=self.n_processes)

    def output(self):
        return [luigi.LocalTarget(self.out_root)]


class ForegroundSegmentation(luigi.Task):
    video = luigi.Parameter()
    end_frame = luigi.IntParameter(default=30000)

    def __init__(self, *args, **kwargs):
        super(ForegroundSegmentation, self).__init__(*args, **kwargs)
        self.crop = GlobalParams().crop

        self.out_root = os.path.join(os.path.splitext(self.video)[0], str(self.end_frame))
        self.video_out = self.out_root + '/rat_sepbg.avi'
        self.img_out = self.out_root + '/images'
        self.mask_out = self.out_root + '/masks'
        self.bg_out = self.out_root + '/bg'

    def run(self):
        foreground_segmentation.run(video_in=self.video,
                                    out_dir=self.out_root,
                                    end_frame=self.end_frame,
                                    crop_bbox_xyxy=self.crop)

    def output(self):
        return [luigi.LocalTarget(self.img_out),
                luigi.LocalTarget(self.mask_out),
                luigi.LocalTarget(self.bg_out)]


class AutoAnnotation(luigi.Task):
    annot_params = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super(AutoAnnotation, self).__init__(*args, **kwargs)

        tmp = 'default' if self.annot_params is None else os.path.splitext(os.path.basename(self.annot_params))[0]
        self.annot_dir = os.path.join(os.path.dirname(self.input()[1].path), 'annots', tmp)

    def requires(self):
        return ForegroundSegmentation()

    def run(self):
        automatic_annotation.run(mask_dir_base=self.input()[1].path,
                                 out_dir_base=self.annot_dir,
                                 params_json=self.annot_params,
                                 no_seq_folder=True)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_out
                luigi.LocalTarget(self.input()[1].path),  # mask_out
                luigi.LocalTarget(self.input()[2].path),  # bg_out
                luigi.LocalTarget(self.annot_dir)]


class Annotation2COCO(luigi.Task):
    aug_params = luigi.Parameter(default=None)
    temporal = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(Annotation2COCO, self).__init__(*args, **kwargs)

        self.exp_name = 'default' if self.aug_params is None else os.path.splitext(os.path.basename(self.aug_params))[0]
        self.aug_json_dir = os.path.join(os.path.dirname(self.input()[0].path), 'annotations', self.exp_name)

    def requires(self):
        return AutoAnnotation()

    def run(self):
        annotation2coco.run(exp_name=self.exp_name,
                            img_dir_base=self.input()[0].path,
                            bg_dir_base=self.input()[2].path,
                            annot_dir_base=self.input()[3].path,
                            out_json_dir=self.aug_json_dir,
                            augment_params=self.aug_params,
                            temporal=self.temporal,
                            no_seq_folder=True)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_out
                luigi.LocalTarget(self.aug_json_dir)]


class PreTrain(luigi.Task):
    model_type = luigi.Parameter(default='instance')
    n_iter = luigi.IntParameter(default=50000)
    val_annot = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super(PreTrain, self).__init__(*args, **kwargs)

        self.annot_ext = '_part.json' if self.model_type == 'part' else '_keypoint.json'
        self.model_dir = os.path.join(os.path.dirname(self.input()[0].path), 'models', self.model_type + '_' + str(self.n_iter)) + '_v000'

    def requires(self):
        return Annotation2COCO()

    def run(self):
        train_annot = os.path.join(self.input()[-1].path, os.path.basename(self.input()[-1].path) + self.annot_ext)
        is_part = (self.model_type == 'part')
        inst_only = not is_part and (self.model_type != 'keypoint')
        train_detectron2.run(train_annot=train_annot,
                             out_dir=self.model_dir,
                             n_iter=self.n_iter,
                             val_annot=self.val_annot,
                             is_part=is_part,
                             inst_only=inst_only)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_out
                luigi.LocalTarget(self.model_dir),
                luigi.LocalTarget(self.model_type)]


class Predict(ParallelTask, GPUTask):  #TODO: include separats model
    model_dir = luigi.Parameter(default='./models/instance_finetuned')
    model_type = luigi.Parameter(default='instance')
    out_dir = luigi.Parameter(default=None)
    use_maskrcnn = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(Predict, self).__init__(*args, **kwargs)
        if self.out_dir is None:
            self.out_dir = self.input()[0].path + '_' + ('separats' if not self.use_maskrcnn else os.path.basename(self.model_dir))
            self.out_segmentation_dir = os.path.join(self.out_dir, ('result_segm_v4.4_col3ep14' if not self.use_maskrcnn else 'eval_results'))

    def requires(self):
        return PrepareVideo()

    def run(self):
        if self.use_maskrcnn:
            is_part = (self.model_type == 'part')
            inst_only = not is_part and (self.model_type != 'keypoint')
            train_detectron2.run(out_dir=self.out_dir,
                                 model_dir=self.model_dir,
                                 is_part=is_part,
                                 inst_only=inst_only,
                                 eval_only=True,
                                 eval_dir=self.input()[0].path,
                                 eval_batch_size=self.batch_size,
                                 n_processes=self.n_processes)
        else:
            import separats_segmentation as separats
            separats.run(input_dir=self.input()[0].path,
                         n_processes=self.n_processes,
                         batch_size=self.batch_size)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_dir
                luigi.LocalTarget(self.out_segmentation_dir)]
                # luigi.LocalTarget(os.path.join(self.out_dir, 'result_segm_v4.4_col3ep14'))]
                # luigi.LocalTarget(os.path.join(self.out_dir, 'eval_results'))]


class OpticalFlow(GPUTask):
    gma_dir = luigi.Parameter(default='./external/GMA')

    def __init__(self, *args, **kwargs):
        super(OpticalFlow, self).__init__(*args, **kwargs)
        seqs = ['seq000']
        if os.path.exists(self.input()[0].path):
            seqs = utils.io.list_directory(self.input()[0].path, only_dirs=True, full_path=False)
        self.out_dir = os.path.join(self.input()[0].path, seqs[-1], 'GMA')

    def requires(self):
        return PrepareVideo()

    def run(self):
        run_gma.run(self.input()[0].path, self.gma_dir, self.batch_size)

    def output(self):
        return luigi.LocalTarget(self.out_dir)


class RunGPUTasks(luigi.Task):
    def __init__(self, *args, **kwargs):
        super(RunGPUTasks, self).__init__(*args, **kwargs)

    def requires(self):
        return Predict(), OpticalFlow()

    def run(self):
        pass

    def output(self):
        return self.input()


class PropagatePredictions(ParallelTask):
    overlap_thrs = luigi.FloatParameter(default=0.1)
    params_json = luigi.Parameter(default=None)
    use_old = luigi.BoolParameter(default=False)
    save_only_scores = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(PropagatePredictions, self).__init__(*args, **kwargs)
        self.out_dir = os.path.join(os.path.dirname(self.input()[0][1].path),
                                    'prop_results_overlap_thrs_{}'.format(self.overlap_thrs))

    def requires(self):
        return RunGPUTasks()

    def run(self):
        iterative_training.propagate_detectron2(model_dir=os.path.dirname(self.input()[0][1].path),
                                                out_dir=self.out_dir,
                                                overlap_thrs=self.overlap_thrs,
                                                params_json=self.params_json,
                                                use_old=self.use_old,
                                                img_base_dir=self.input()[0][0].path,
                                                n_processes=self.n_processes,
                                                chunksize=self.chunksize,
                                                save_only_scores=self.save_only_scores)

    def output(self):
        return [luigi.LocalTarget(self.input()[0][0].path),  # img_dir
                luigi.LocalTarget(self.out_dir),
                luigi.LocalTarget(os.path.join(self.out_dir, 'seq015'))]


class EvaluatePredictions(luigi.Task):
    gt_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(EvaluatePredictions, self).__init__(*args, **kwargs)

    def requires(self):
        return PropagatePredictions()

    def run(self):
        iterative_training.evaluate_results(self.input()[1].path, self.gt_dir)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_dir
                luigi.LocalTarget(self.input()[1].path),  # out_dir
                luigi.LocalTarget(os.path.join(self.input()[1].path, 'eval.json'))]


class MatchPredictions(ParallelTask):
    guide_dir = luigi.Parameter(default=None)

    def __init__(self, *args, **kwargs):
        super(MatchPredictions, self).__init__(*args, **kwargs)
        self.out_dir = self.input()[1].path + '_matched'
        if self.guide_dir is None:
            self.guide_dir = self.input()[1].path

    def requires(self):
        return PropagatePredictions()

    def run(self):
        propagate_instances.match_labels(self.input()[1].path, self.guide_dir, self.out_dir,
                                         n_processes=self.n_processes)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),  # img_dir
                luigi.LocalTarget(self.out_dir)]


class BehaviorEstimation(ParallelTask):
    config_file = luigi.Parameter(default='./config.arch')
    instances_dir = luigi.Parameter(default=None)
    match_predictions = luigi.BoolParameter(default=False)
    merge_outputs = luigi.BoolParameter(default=True)

    def __init__(self, *args, **kwargs):
        super(BehaviorEstimation, self).__init__(*args, **kwargs)
        if self.instances_dir is None:
            self.instances_dir = self.input()[1].path
        self.output_dir = self.instances_dir + '_behaviors'

    def requires(self):
        if self.match_predictions:
            return MatchPredictions()
        else:
            return PropagatePredictions()

    def run(self):
        behavior.run(instances_dir=self.instances_dir, base_arch_file=self.config_file,
                     output_dir=self.output_dir, eval=False, merge=self.merge_outputs, verbose=3,
                     n_processes=self.n_processes)

    def output(self):
        return [luigi.LocalTarget(self.output_dir),         # Behaviors
                luigi.LocalTarget(self.input()[1].path),    # Predictions
                luigi.LocalTarget(self.input()[0].path)]    # Images


class MatchSequences(ParallelTask):
    def __init__(self, *args, **kwargs):
        super(MatchSequences, self).__init__(*args, **kwargs)
        self.switched_behaviors = os.path.join(self.input()[0].path, 'behaviors_dt_switched.arch')
        self.switched_sequences = os.path.join(self.input()[0].path, 'switched_sequences.json')

    def requires(self):
        return BehaviorEstimation()

    def run(self):
        switch_up_behaviors.match_behaviors_between_sequences(self.input()[1].path,
                                                              self.input()[0].path)

    def output(self):
        return [luigi.LocalTarget(self.switched_behaviors),     # Behaviors
                luigi.LocalTarget(self.input()[1].path),        # Predictions
                luigi.LocalTarget(self.switched_sequences),     # Matched sequences
                luigi.LocalTarget(self.input()[-1].path)]       # Images


class CalculateStatistics(ParallelTask):
    inner_box_coordinates = luigi.ListParameter(default=[213, 12, 408, 402])
    inner_box_size = luigi.ListParameter(default=[0.4, 0.8])
    use_absolute_coordinates = luigi.BoolParameter(default=False)
    fn_suffix = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(CalculateStatistics, self).__init__(*args, **kwargs)
        self.out_fn = self.fn_suffix.join(os.path.splitext(f"{self.input()[1][1].path}_statistics.xlsx"))
        self.crop = GlobalParams().crop

    def requires(self):
        return PrepareVideo(), MatchSequences()

    def run(self):
        if self.use_absolute_coordinates:
            box_coords = np.asarray(self.inner_box_coordinates, dtype=int)
            crop_coords = np.asarray(self.crop, dtype=int)
            box_coords[:2] -= crop_coords[:2]
            box_coords[2:] -= crop_coords[:2]
            resize_ratio_x = (crop_coords[2] - crop_coords[0]) / 640
            resize_ratio_y = (crop_coords[3] - crop_coords[1]) / 420
            box_coords[::2] = (box_coords[::2] / resize_ratio_x).astype(int)
            box_coords[1::2] = (box_coords[1::2] / resize_ratio_y).astype(int)
            self.inner_box_coordinates = box_coords.tolist()
            print(f"[NOTE] Relative inner-box-coordinates: {self.inner_box_coordinates}")
        position_statistics.run(self.input()[1][1].path, switched_sequences=self.input()[1][2].path,
                                inner_box_xyxy=self.inner_box_coordinates, inner_box_size_in_meters=self.inner_box_size,
                                n_processes=self.n_processes, chunksize=self.chunksize,
                                img_dir_base=self.input()[0][0].path, fn_suffix=self.fn_suffix)

    def output(self):
        return [luigi.LocalTarget(self.input()[1][0].path),  # Behaviors
                luigi.LocalTarget(self.input()[1][1].path),  # Predictions
                luigi.LocalTarget(self.out_fn)]              # Statistics


class VisualizeResults(ParallelTask):
    fn_suffix = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(VisualizeResults, self).__init__(*args, **kwargs)
        self.out_fn = f"{self.input()[1].path}{self.fn_suffix}.mp4"

    def requires(self):
        return MatchSequences()

    def run(self):
        visualization.visualize_predictions_memory_conservative(self.input()[-1].path, self.input()[1].path,
                                                                switched_sequences=self.input()[2].path,
                                                                concat_videos=True,
                                                                n_processes=self.n_processes,
                                                                out_fn=self.out_fn)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),    # Behaviors
                luigi.LocalTarget(self.input()[1].path),    # Predictions
                luigi.LocalTarget(self.out_fn)]             # Video


class EstimateConfidenceScores(ParallelTask):
    def __init__(self, *args, **kwargs):
        super(EstimateConfidenceScores, self).__init__(*args, **kwargs)
        self.out_fn = "_marked".join(os.path.splitext(self.input()[0].path))

    def requires(self):
        return CalculateStatistics()

    def run(self):
        confidence.run(self.input()[1].path, self.input()[0].path, out_file=self.out_fn)

    def output(self):
        return [luigi.LocalTarget(self.out_fn),              # Behaviors marked
                luigi.LocalTarget(self.input()[1].path),     # Predictions
                luigi.LocalTarget(self.input()[2].path)]     # Statistics


class FixSequences(ParallelTask):
    def __init__(self, *args, **kwargs):
        super(FixSequences, self).__init__(*args, **kwargs)
        self.out_fn = "_marked".join(os.path.splitext(self.input()[2].path))  # self.input()[2].path.replace(".arch", ".txt")

    def requires(self):
        return EstimateConfidenceScores()

    def run(self):
        fix_sequences.run(self.input()[0].path, self.input()[1].path)
        luigi.build([CalculateStatistics(fn_suffix="_marked")])  # , VisualizeResults(fn_suffix="_marked")
        # utils.io.save(self.out_fn, "Nothing to see here. It's just a dummy file to indicate that it's FINISHED :P")

    def output(self):
        return [luigi.LocalTarget(self.out_fn)]                 # FINISHED


class RunAllTasks(luigi.Task):
    def __init__(self, *args, **kwargs):
        super(RunAllTasks, self).__init__(*args, **kwargs)

    def requires(self):
        # return CalculateStatistics(), \
        return EstimateConfidenceScores(), \
               VisualizeResults()

    def run(self):
        pass

    def output(self):
        return self.input()


class BehaviorEvaluation(luigi.Task):
    config_file = luigi.Parameter()
    gt_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(BehaviorEvaluation, self).__init__(*args, **kwargs)
        self.output_dir = self.input()[0].path

    def requires(self):
        return BehaviorEstimation(config_file=self.config_file, match_predictions=True, merge_outputs=False)

    def run(self):
        behavior.run(base_arch_file=self.config_file, output_dir=self.output_dir,
                     eval=True, behaviors_gt_dir=self.gt_dir,
                     behaviors_dt_dir=os.path.join(self.output_dir, 'behaviors_dt'),
                     verbose=3)

    def output(self):
        return [luigi.LocalTarget(os.path.join(self.output_dir, 'classification_report.txt'))]


class ConvertGT(luigi.Task):
    annot_dir = luigi.Parameter()  # /home/lkopi/repo/rat_annotation/done/vis/train
    img_dir = luigi.Parameter()  # /media/hdd2/lkopi/datasets/rats/tracking_test/test_videos/training_data/images
    root_dir = luigi.Parameter(default=None)
    version = luigi.IntParameter(default=1)

    def __init__(self, *args, **kwargs):
        super(ConvertGT, self).__init__(*args, **kwargs)

        if self.root_dir is None:
            self.root_dir = os.path.dirname(self.img_dir)
        self.out_dir = os.path.join(self.root_dir, 'test_set', 'v{:03d}'.format(self.version))

    def run(self):
        handannot2coco.run(img_dir=self.img_dir, annot_dir=self.annot_dir, out_dir=self.out_dir)

    def output(self):
        out_img_dir = os.path.join(self.out_dir, 'images')
        out_mask_dir = os.path.join(self.out_dir, 'masks')
        out_annot_json = os.path.join(self.out_dir, 'annotations', 'test.json')
        return [luigi.LocalTarget(out_img_dir),
                luigi.LocalTarget(out_mask_dir),
                luigi.LocalTarget(out_annot_json)]


class TuneModel(luigi.Task):
    model_dir = luigi.Parameter(default=None)
    n_iter = luigi.IntParameter(default=5000)  # TODO make it an additional 5000 instead of max 5000
    version = luigi.IntParameter(default=1)

    def __init__(self, *args, **kwargs):
        super(TuneModel, self).__init__(*args, **kwargs)
        assert self.model_dir is not None or 0 < self.version

        root_dir = os.path.dirname(os.path.dirname(self.input()[0].path))
        if self.model_dir is None:
            self.model_dir = os.path.join(root_dir, 'models', 'v{:03d}'.format(self.version - 1))
        self.out_dir = '{}{:03d}'.format(self.model_dir[:-3], self.version)

    def requires(self):
        return ConvertGT(version=self.version)

    def run(self):
        # Init: copy the previous model
        utils.io.make_directory(self.out_dir)
        if os.path.exists(self.model_dir):
            os.system("cp -r {}/* {}/".format(self.model_dir, self.out_dir))

        # Fine-tune it
        train_detectron2.run(train_annot=self.input()[-1].path,
                             out_dir=self.out_dir,
                             n_iter=self.n_iter,
                             is_part=False,
                             inst_only=True,
                             resume=True)

    def output(self):
        return [luigi.LocalTarget(self.input()[0].path),
                luigi.LocalTarget(self.out_dir)]
