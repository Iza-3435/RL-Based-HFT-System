import asyncio
import time
import logging
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from enum import Enum
import json
import sys
import gc
logger = logging.getLogger(__name__)
class MetricType(Enum):

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE = "resource"
    ACCURACY = "accuracy"
    NETWORK = "network"
    SYSTEM = "system"
@dataclass
class PerformanceMetric:

    timestamp: float
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class BenchmarkResult:

    test_name: str
    timestamp: float
    duration_seconds: float
    target_metric: str
    target_value: float
    actual_value: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
@dataclass
class AlertRule:

    metric_name: str
    condition: str
    threshold: float
    window_size: int = 10
    consecutive_violations: int = 3
    callback: Optional[Callable] = None
class PerformanceTracker:


    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.start_time = time.time()

        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_definitions: Dict[str, Dict] = {}

        self.baselines: Dict[str, float] = {}
        self.optimization_targets: Dict[str, float] = {}

        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []

        self.benchmark_results: List[BenchmarkResult] = []

        self.system_monitor_active = False
        self.system_monitor_thread = None

        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        self.optimization_history: List[Dict] = []

        logger.info("PerformanceTracker initialized")

    def define_metric(self, name: str, metric_type: MetricType, unit: str,
                     description: str = "", target_value: Optional[float] = None):

        self.metric_definitions[name] = {
            'type': metric_type,
            'unit': unit,
            'description': description,
            'target_value': target_value,
            'created_at': time.time()
        }

        if target_value is not None:
            self.optimization_targets[name] = target_value

        logger.info(f"Defined metric: {name} ({metric_type.value}, {unit})")

    def record_metric(self, name: str, value: float, component: str = "system",
                     metadata: Dict[str, Any] = None):

        timestamp = time.time()

        definition = self.metric_definitions.get(name, {})
        metric_type = definition.get('type', MetricType.SYSTEM)
        unit = definition.get('unit', 'units')

        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            component=component,
            metadata=metadata or {}
        )

        self.metrics[name].append(metric)

        self.component_metrics[component][name] = metric

        self._check_alert_rules(name, value)

        return metric

    def record_latency(self, operation: str, latency_ms: float, component: str = "system"):

        return self.record_metric(
            f"{operation}_latency",
            latency_ms,
            component,
            {'operation': operation, 'measurement_type': 'latency'}
        )

    def record_throughput(self, operation: str, rate_per_second: float, component: str = "system"):

        return self.record_metric(
            f"{operation}_throughput",
            rate_per_second,
            component,
            {'operation': operation, 'measurement_type': 'throughput'}
        )

    def start_system_monitoring(self):

        if self.system_monitor_active:
            logger.warning("System monitoring already active")
            return

        self.system_monitor_active = True
        self.system_monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
        self.system_monitor_thread.start()

        logger.info("Started system monitoring")

    def stop_system_monitoring(self):

        self.system_monitor_active = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)

        logger.info("Stopped system monitoring")

    def _system_monitor_loop(self):

        while self.system_monitor_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                self.record_metric("cpu_usage_percent", cpu_percent, "system")

                memory = psutil.virtual_memory()
                self.record_metric("memory_usage_percent", memory.percent, "system")
                self.record_metric("memory_available_gb", memory.available / (1024**3), "system")

                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_metric("disk_read_mb_per_sec", disk_io.read_bytes / (1024**2), "system")
                    self.record_metric("disk_write_mb_per_sec", disk_io.write_bytes / (1024**2), "system")

                network_io = psutil.net_io_counters()
                if network_io:
                    self.record_metric("network_bytes_sent_per_sec", network_io.bytes_sent, "system")
                    self.record_metric("network_bytes_recv_per_sec", network_io.bytes_recv, "system")

                process = psutil.Process()
                self.record_metric("process_memory_rss_mb", process.memory_info().rss / (1024**2), "process")
                self.record_metric("process_cpu_percent", process.cpu_percent(), "process")

                gc_counts = gc.get_count()
                self.record_metric("gc_objects_gen0", gc_counts[0], "python")
                self.record_metric("gc_objects_gen1", gc_counts[1], "python")
                self.record_metric("gc_objects_gen2", gc_counts[2], "python")

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def add_alert_rule(self, metric_name: str, condition: str, threshold: float,
                      window_size: int = 10, consecutive_violations: int = 3,
                      callback: Optional[Callable] = None):

        rule = AlertRule(
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            window_size=window_size,
            consecutive_violations=consecutive_violations,
            callback=callback
        )

        self.alert_rules[metric_name] = rule
        logger.info(f"Added alert rule: {metric_name} {condition} {threshold}")

    def _check_alert_rules(self, metric_name: str, value: float):

        if metric_name not in self.alert_rules:
            return

        rule = self.alert_rules[metric_name]
        recent_metrics = list(self.metrics[metric_name])[-rule.window_size:]

        if len(recent_metrics) < rule.consecutive_violations:
            return

        violations = 0
        for metric in recent_metrics[-rule.consecutive_violations:]:
            violation = False

            if rule.condition == "greater_than" and metric.value > rule.threshold:
                violation = True
            elif rule.condition == "less_than" and metric.value < rule.threshold:
                violation = True
            elif rule.condition == "equals" and abs(metric.value - rule.threshold) < 1e-6:
                violation = True
            elif rule.condition == "not_equals" and abs(metric.value - rule.threshold) >= 1e-6:
                violation = True

            if violation:
                violations += 1

        if violations >= rule.consecutive_violations:
            self._trigger_alert(rule, value, recent_metrics)

    def _trigger_alert(self, rule: AlertRule, current_value: float, recent_metrics: List[PerformanceMetric]):

        alert = {
            'timestamp': time.time(),
            'metric_name': rule.metric_name,
            'condition': rule.condition,
            'threshold': rule.threshold,
            'current_value': current_value,
            'consecutive_violations': rule.consecutive_violations,
            'recent_values': [m.value for m in recent_metrics[-5:]]
        }

        self.alert_history.append(alert)

        logger.warning(f"PERFORMANCE ALERT: {rule.metric_name} {rule.condition} {rule.threshold}, "
                      f"current: {current_value}")

        if rule.callback:
            try:
                rule.callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in general alert callback: {e}")

    def set_baseline(self, metric_name: str, value: Optional[float] = None):

        if value is None:
            if metric_name in self.metrics and self.metrics[metric_name]:
                values = [m.value for m in self.metrics[metric_name]]
                value = np.mean(values)
            else:
                logger.warning(f"No data available for baseline: {metric_name}")
                return

        self.baselines[metric_name] = value
        logger.info(f"Set baseline for {metric_name}: {value}")

    def measure_optimization_impact(self, optimization_name: str,
                                  metrics_to_track: List[str],
                                  baseline_window: int = 100,
                                  measurement_window: int = 100) -> Dict[str, Any]:

        results = {
            'optimization_name': optimization_name,
            'timestamp': time.time(),
            'metrics_impact': {},
            'overall_improvement': 0.0
        }

        for metric_name in metrics_to_track:
            if metric_name not in self.metrics:
                continue

            all_metrics = list(self.metrics[metric_name])

            if len(all_metrics) < baseline_window + measurement_window:
                logger.warning(f"Insufficient data for impact analysis: {metric_name}")
                continue

            baseline_metrics = all_metrics[-(baseline_window + measurement_window):-measurement_window]
            current_metrics = all_metrics[-measurement_window:]

            baseline_values = [m.value for m in baseline_metrics]
            current_values = [m.value for m in current_metrics]

            baseline_mean = np.mean(baseline_values)
            current_mean = np.mean(current_values)

            definition = self.metric_definitions.get(metric_name, {})
            metric_type = definition.get('type', MetricType.SYSTEM)

            if metric_type in [MetricType.LATENCY]:
                improvement_pct = ((baseline_mean - current_mean) / baseline_mean) * 100
            else:
                improvement_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

            from scipy import stats
            statistic, p_value = stats.ttest_ind(baseline_values, current_values)

            results['metrics_impact'][metric_name] = {
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'improvement_percent': improvement_pct,
                'p_value': p_value,
                'statistically_significant': p_value < 0.05,
                'baseline_std': np.std(baseline_values),
                'current_std': np.std(current_values)
            }

        significant_improvements = [
            impact['improvement_percent']
            for impact in results['metrics_impact'].values()
            if impact['statistically_significant'] and impact['improvement_percent'] > 0
        ]

        if significant_improvements:
            results['overall_improvement'] = np.mean(significant_improvements)

        self.optimization_history.append(results)

        logger.info(f"Optimization impact analysis for '{optimization_name}': "
                   f"{results['overall_improvement']:.2f}% improvement")

        return results

    def run_benchmark(self, test_name: str, test_function: Callable,
                     target_metric: str, target_value: float,
                     iterations: int = 1, **kwargs) -> BenchmarkResult:

        logger.info(f"Running benchmark: {test_name}")

        start_time = time.time()
        results = []

        for i in range(iterations):
            try:
                iteration_start = time.time()
                result = test_function(**kwargs)
                iteration_time = time.time() - iteration_start

                results.append({
                    'iteration': i + 1,
                    'result': result,
                    'duration': iteration_time
                })

            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                results.append({
                    'iteration': i + 1,
                    'result': None,
                    'duration': 0,
                    'error': str(e)
                })

        total_duration = time.time() - start_time

        successful_results = [r for r in results if r.get('result') is not None]

        if successful_results:
            if target_metric == 'duration':
                actual_value = np.mean([r['duration'] for r in successful_results])
            elif target_metric == 'throughput':
                avg_duration = np.mean([r['duration'] for r in successful_results])
                actual_value = 1.0 / avg_duration if avg_duration > 0 else 0
            else:
                sample_result = successful_results[0]['result']
                if isinstance(sample_result, dict) and target_metric in sample_result:
                    actual_value = np.mean([r['result'][target_metric] for r in successful_results])
                else:
                    actual_value = sample_result if isinstance(sample_result, (int, float)) else 0
        else:
            actual_value = 0

        if target_metric in ['latency', 'duration']:
            success = actual_value <= target_value
        else:
            success = actual_value >= target_value

        benchmark_result = BenchmarkResult(
            test_name=test_name,
            timestamp=start_time,
            duration_seconds=total_duration,
            target_metric=target_metric,
            target_value=target_value,
            actual_value=actual_value,
            success=success,
            details={
                'iterations': iterations,
                'successful_iterations': len(successful_results),
                'results': results
            }
        )

        self.benchmark_results.append(benchmark_result)

        logger.info(f"Benchmark '{test_name}' completed: {actual_value:.4f} vs target {target_value} "
                   f"({'PASS' if success else 'FAIL'})")

        return benchmark_result

    def get_latency_distribution(self, metric_name: str, window_minutes: int = 60) -> Dict[str, float]:

        cutoff_time = time.time() - (window_minutes * 60)

        if metric_name not in self.metrics:
            return {}

        recent_metrics = [
            m for m in self.metrics[metric_name]
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        values = [m.value for m in recent_metrics]

        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p90': np.percentile(values, 90),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'p99_9': np.percentile(values, 99.9)
        }

    def get_throughput_analysis(self, metric_name: str, window_minutes: int = 60) -> Dict[str, float]:

        cutoff_time = time.time() - (window_minutes * 60)

        if metric_name not in self.metrics:
            return {}

        recent_metrics = [
            m for m in self.metrics[metric_name]
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]

        if len(values) > 1:
            time_diffs = np.diff(timestamps)
            value_diffs = np.diff(values)
            rates = value_diffs / time_diffs
            avg_rate = np.mean(rates[rates > 0]) if len(rates[rates > 0]) > 0 else 0
        else:
            avg_rate = 0

        return {
            'current_throughput': values[-1] if values else 0,
            'average_throughput': np.mean(values),
            'peak_throughput': np.max(values),
            'throughput_volatility': np.std(values),
            'average_rate_change': avg_rate,
            'samples': len(values)
        }

    def get_resource_utilization_summary(self, window_minutes: int = 60) -> Dict[str, Any]:

        cutoff_time = time.time() - (window_minutes * 60)

        resource_metrics = [
            'cpu_usage_percent',
            'memory_usage_percent',
            'process_memory_rss_mb',
            'process_cpu_percent'
        ]

        summary = {}

        for metric_name in resource_metrics:
            if metric_name in self.metrics:
                recent_metrics = [
                    m for m in self.metrics[metric_name]
                    if m.timestamp >= cutoff_time
                ]

                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    summary[metric_name] = {
                        'current': values[-1],
                        'average': np.mean(values),
                        'peak': np.max(values),
                        'trend': 'increasing' if values[-1] > np.mean(values) else 'decreasing'
                    }

        return summary

    def detect_performance_anomalies(self, metric_name: str,
                                   sensitivity: float = 2.0,
                                   window_size: int = 100) -> List[Dict[str, Any]]:

        if metric_name not in self.metrics:
            return []

        recent_metrics = list(self.metrics[metric_name])[-window_size:]

        if len(recent_metrics) < 10:
            return []

        values = np.array([m.value for m in recent_metrics])
        timestamps = [m.timestamp for m in recent_metrics]

        window = min(20, len(values) // 2)
        anomalies = []

        for i in range(window, len(values)):
            baseline = values[i-window:i]
            current_value = values[i]

            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)

            if baseline_std > 0:
                z_score = abs((current_value - baseline_mean) / baseline_std)

                if z_score > sensitivity:
                    anomalies.append({
                        'timestamp': timestamps[i],
                        'value': current_value,
                        'baseline_mean': baseline_mean,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3.0 else 'medium',
                        'type': 'spike' if current_value > baseline_mean else 'drop'
                    })

        return anomalies

    def get_optimization_effectiveness_report(self) -> Dict[str, Any]:

        if not self.optimization_history:
            return {'message': 'No optimization history available'}

        report = {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': 0,
            'average_improvement': 0.0,
            'best_optimization': None,
            'worst_optimization': None,
            'optimization_timeline': [],
            'metric_improvements': defaultdict(list)
        }

        improvements = []

        for opt in self.optimization_history:
            improvement = opt.get('overall_improvement', 0)
            improvements.append(improvement)

            if improvement > 0:
                report['successful_optimizations'] += 1

            report['optimization_timeline'].append({
                'name': opt['optimization_name'],
                'timestamp': opt['timestamp'],
                'improvement': improvement
            })

            for metric_name, impact in opt.get('metrics_impact', {}).items():
                if impact.get('statistically_significant', False):
                    report['metric_improvements'][metric_name].append(impact['improvement_percent'])

        if improvements:
            report['average_improvement'] = np.mean([imp for imp in improvements if imp > 0])

            best_idx = np.argmax(improvements)
            worst_idx = np.argmin(improvements)

            report['best_optimization'] = {
                'name': self.optimization_history[best_idx]['optimization_name'],
                'improvement': improvements[best_idx]
            }

            report['worst_optimization'] = {
                'name': self.optimization_history[worst_idx]['optimization_name'],
                'improvement': improvements[worst_idx]
            }

        cumulative_improvement = 0
        for opt in self.optimization_history:
            cumulative_improvement += opt.get('overall_improvement', 0)

        report['cumulative_improvement'] = cumulative_improvement

        return report

    def export_performance_data(self, filename: str = None,
                              metrics: List[str] = None,
                              hours_back: int = 24) -> str:

        if not filename:
            filename = f"performance_data_{int(time.time())}.csv"

        cutoff_time = time.time() - (hours_back * 3600)

        metrics_to_export = metrics or list(self.metrics.keys())

        data_rows = []

        for metric_name in metrics_to_export:
            if metric_name not in self.metrics:
                continue

            for metric in self.metrics[metric_name]:
                if metric.timestamp >= cutoff_time:
                    row = {
                        'timestamp': metric.timestamp,
                        'datetime': datetime.fromtimestamp(metric.timestamp).isoformat(),
                        'metric_name': metric.metric_name,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'unit': metric.unit,
                        'component': metric.component
                    }

                    for key, value in metric.metadata.items():
                        row[f'meta_{key}'] = value

                    data_rows.append(row)

        if data_rows:
            df = pd.DataFrame(data_rows)
            df = df.sort_values('timestamp')
            df.to_csv(filename, index=False)

            logger.info(f"Exported {len(data_rows)} performance records to {filename}")
        else:
            logger.warning("No performance data to export")

        return filename

    def get_system_health_score(self) -> Dict[str, Any]:

        health_score = {
            'overall_score': 100.0,
            'component_scores': {},
            'issues': [],
            'recommendations': []
        }

        recent_alerts = [
            alert for alert in self.alert_history
            if time.time() - alert['timestamp'] < 3600
        ]

        if recent_alerts:
            alert_penalty = min(50, len(recent_alerts) * 10)
            health_score['overall_score'] -= alert_penalty
            health_score['issues'].append(f"{len(recent_alerts)} alerts in the last hour")

        resource_summary = self.get_resource_utilization_summary(window_minutes=10)

        for metric_name, stats in resource_summary.items():
            current = stats.get('current', 0)

            if 'cpu' in metric_name and current > 80:
                penalty = (current - 80) * 0.5
                health_score['overall_score'] -= penalty
                health_score['issues'].append(f"High CPU usage: {current:.1f}%")

            elif 'memory' in metric_name and current > 85:
                penalty = (current - 85) * 0.8
                health_score['overall_score'] -= penalty
                health_score['issues'].append(f"High memory usage: {current:.1f}%")

        for metric_name in self.metrics:
            if 'latency' in metric_name:
                distribution = self.get_latency_distribution(metric_name, window_minutes=10)

                if distribution and distribution.get('p95', 0) > 1000:
                    health_score['overall_score'] -= 15
                    health_score['issues'].append(f"High latency in {metric_name}: P95 = {distribution['p95']:.1f}ms")

        if health_score['overall_score'] < 90:
            if any('cpu' in issue.lower() for issue in health_score['issues']):
                health_score['recommendations'].append("Consider optimizing CPU-intensive operations")

            if any('memory' in issue.lower() for issue in health_score['issues']):
                health_score['recommendations'].append("Review memory usage and consider garbage collection tuning")

            if any('latency' in issue.lower() for issue in health_score['issues']):
                health_score['recommendations'].append("Investigate latency bottlenecks and network issues")

        health_score['overall_score'] = max(0, health_score['overall_score'])

        return health_score

    def get_comprehensive_report(self) -> Dict[str, Any]:

        runtime = time.time() - self.start_time

        report = {
            'report_timestamp': time.time(),
            'system_runtime_hours': runtime / 3600,
            'total_metrics_tracked': len(self.metrics),
            'total_measurements': sum(len(metrics) for metrics in self.metrics.values()),

            'system_health': self.get_system_health_score(),
            'recent_alerts': len([a for a in self.alert_history if time.time() - a['timestamp'] < 3600]),

            'resource_utilization': self.get_resource_utilization_summary(),

            'optimization_effectiveness': self.get_optimization_effectiveness_report(),

            'recent_benchmarks': [
                {
                    'name': b.test_name,
                    'success': b.success,
                    'target_value': b.target_value,
                    'actual_value': b.actual_value
                }
                for b in self.benchmark_results[-10:]
            ],

            'key_metrics': {}
        }

        important_metrics = [name for name in self.metrics.keys()
                           if any(keyword in name.lower()
                                 for keyword in ['latency', 'throughput', 'cpu', 'memory'])]

        for metric_name in important_metrics[:10]:
            if 'latency' in metric_name:
                report['key_metrics'][metric_name] = self.get_latency_distribution(metric_name, 60)
            elif 'throughput' in metric_name:
                report['key_metrics'][metric_name] = self.get_throughput_analysis(metric_name, 60)
            else:
                recent_values = [m.value for m in list(self.metrics[metric_name])[-100:]]
                if recent_values:
                    report['key_metrics'][metric_name] = {
                        'current': recent_values[-1],
                        'average': np.mean(recent_values),
                        'trend': 'up' if recent_values[-1] > np.mean(recent_values[:-10]) else 'down'
                    }

        return report

    def cleanup_old_data(self, days_to_keep: int = 7):

        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        cleaned_count = 0

        for metric_name in list(self.metrics.keys()):
            original_len = len(self.metrics[metric_name])

            self.metrics[metric_name] = deque(
                (m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time),
                maxlen=self.metrics[metric_name].maxlen
            )

            cleaned_count += original_len - len(self.metrics[metric_name])

        self.alert_history = deque(
            (alert for alert in self.alert_history if alert['timestamp'] >= cutoff_time),
            maxlen=self.alert_history.maxlen
        )

        self.benchmark_results = [
            b for b in self.benchmark_results
            if b.timestamp >= cutoff_time
        ]

        logger.info(f"Cleaned up {cleaned_count} old performance records")
async def test_performance_tracker():


    tracker = PerformanceTracker(monitoring_interval=0.5)

    print("Testing PerformanceTracker...")

    tracker.define_metric("api_latency", MetricType.LATENCY, "ms", "API response latency", 100.0)
    tracker.define_metric("order_throughput", MetricType.THROUGHPUT, "orders/sec", "Order processing rate", 1000.0)
    tracker.define_metric("cpu_usage", MetricType.RESOURCE, "percent", "CPU utilization")

    tracker.start_system_monitoring()

    import random

    for i in range(50):
        base_latency = 80 + random.uniform(-20, 20)
        if i % 10 == 0:
            base_latency += random.uniform(50, 200)

        tracker.record_latency("api_request", base_latency, "api_server")

        throughput = 900 + random.uniform(-100, 200)
        tracker.record_throughput("order_processing", throughput, "order_engine")

        tracker.record_metric("custom_score", random.uniform(0, 100), "custom_component")

        await asyncio.sleep(0.1)

    tracker.add_alert_rule("api_latency", "greater_than", 150.0, window_size=5, consecutive_violations=3)

    tracker.set_baseline("api_latency")
    tracker.set_baseline("order_throughput")

    def dummy_operation():
        time.sleep(0.01)
        return {"processed": True, "duration": 0.01}

    benchmark_result = tracker.run_benchmark(
        "dummy_operation_test",
        dummy_operation,
        "duration",
        0.015,
        iterations=5
    )

    print(f"Benchmark result: {benchmark_result.success} (target: {benchmark_result.target_value}, actual: {benchmark_result.actual_value:.4f})")

    print("\nSimulating optimization...")

    for i in range(30):
        improved_latency = 60 + random.uniform(-15, 15)
        tracker.record_latency("api_request", improved_latency, "api_server")
        await asyncio.sleep(0.05)

    impact = tracker.measure_optimization_impact(
        "API Caching Implementation",
        ["api_latency"],
        baseline_window=50,
        measurement_window=30
    )

    print(f"Optimization impact: {impact['overall_improvement']:.2f}% improvement")

    latency_dist = tracker.get_latency_distribution("api_latency", window_minutes=5)
    print(f"\nLatency distribution:")
    for percentile, value in latency_dist.items():
        print(f"  {percentile}: {value:.2f}")

    health = tracker.get_system_health_score()
    print(f"\nSystem health score: {health['overall_score']:.1f}/100")
    if health['issues']:
        print("Issues:", health['issues'])

    report = tracker.get_comprehensive_report()
    print(f"\nComprehensive report generated:")
    print(f"  Runtime: {report['system_runtime_hours']:.2f} hours")
    print(f"  Total measurements: {report['total_measurements']}")
    print(f"  Recent alerts: {report['recent_alerts']}")

    export_file = tracker.export_performance_data(hours_back=1)
    print(f"Performance data exported to: {export_file}")

    tracker.stop_system_monitoring()

    return tracker
if __name__ == "__main__":
    asyncio.run(test_performance_tracker())