import pandas as pd
from typing import List, Tuple
import boto3
from smdebug.profiler.analysis.notebook_utils.training_job import TrainingJob
from smdebug.profiler.analysis.utils.profiler_data_to_pandas import PandasFrame
from smdebug.profiler.analysis.utils.pandas_data_analysis import (
    PandasFrameAnalysis,
    StatsBy,
    Resource,
)

def get_profile_report_folder(bucket, prefix, maxkeys=1):
    s3 = boto3.client("s3")
    results = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=maxkeys)
    filename = results['Contents'][0]['Key']
    profile_report_folder = [s for s in filename.split('/') if "ProfilerReport" in s][0]
    return profile_report_folder


def get_profiling_df(smdebug_trnjob: TrainingJob) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pf = PandasFrame(smdebug_trnjob.profiler_s3_output_path)
    system_metrics_df = pf.get_all_system_metrics()    
    framework_metrics_df = pf.get_all_framework_metrics(
        selected_framework_metrics=["Step:ModeKeys.TRAIN", "Step:ModeKeys.GLOBAL"]
    )    
    return system_metrics_df, framework_metrics_df


def plot_profiling_metrics(smdebug_trnjob: TrainingJob, 
                           dimensions: List[str]=["CPU", "GPU", "I/O"], events: List[str]=["total"], 
                           plot_metrics_histogram: bool=True, plot_heatmap: bool=True, 
                           plot_step_histogram: bool=True, plot_timeline_charts: bool=True,
                           plot_step_timeline_chart: bool=True):
    from smdebug.profiler.analysis.notebook_utils.metrics_histogram import MetricsHistogram
    from smdebug.profiler.analysis.notebook_utils.step_timeline_chart import StepTimelineChart
    from smdebug.profiler.analysis.notebook_utils.step_histogram import StepHistogram
    from smdebug.profiler.analysis.notebook_utils.timeline_charts import TimelineCharts
    from smdebug.profiler.analysis.notebook_utils.heatmap import Heatmap
    
    # Get the metrics reader
    system_metrics_reader = smdebug_trnjob.get_systems_metrics_reader()
    framework_metrics_reader = smdebug_trnjob.get_framework_metrics_reader()

    # Refresh the event file list
    system_metrics_reader.refresh_event_file_list()
    framework_metrics_reader.refresh_event_file_list()

    if plot_metrics_histogram:
        metrics_histogram = MetricsHistogram(system_metrics_reader)
        metrics_histogram.plot(
            starttime=0, 
            endtime=system_metrics_reader.get_timestamp_of_latest_available_file(), 
            select_dimensions=dimensions,
            select_events=events
        )
    
    if plot_heatmap:
        view_heatmap = Heatmap(
            system_metrics_reader,
            framework_metrics_reader,
            select_dimensions=dimensions,
            select_events=events,
            plot_height=200
        )    
    
    if plot_step_histogram:
        step_histogram = StepHistogram(framework_metrics_reader)
        step_histogram.plot(
            starttime=step_histogram.last_timestamp - 5 * 1000 * 1000, 
            endtime=step_histogram.last_timestamp, 
            show_workers=True
        )  
    
    if plot_timeline_charts:
        view_timeline_charts = TimelineCharts(
            system_metrics_reader, 
            framework_metrics_reader,
            select_dimensions=dimensions,
            select_events=events
        )    
    
    if plot_step_timeline_chart:
        view_step_timeline_chart = StepTimelineChart(framework_metrics_reader)    