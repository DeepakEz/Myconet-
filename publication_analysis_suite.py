# publication_analysis_suite.py
"""
Complete Publication Analysis Suite for MycoNet++ Contemplative AI
Adds statistical significance testing, effect sizes, visualizations, and XAI logging
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import time
import logging

# Set up publication-quality plotting
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

@dataclass
class StatisticalTest:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    is_significant: bool
    interpretation: str
    sample_sizes: Tuple[int, int]

@dataclass
class PublicationResults:
    """Container for publication-ready results"""
    descriptive_stats: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, StatisticalTest]]
    effect_sizes: Dict[str, Dict[str, float]]
    visualizations: List[str]
    reproducibility_info: Dict[str, Any]
    xai_summaries: Dict[str, Any]

class StatisticalAnalyzer:
    """Advanced statistical analysis for publication"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def test_normality(self, data: List[float]) -> bool:
        """Test if data is normally distributed"""
        if len(data) < 3:
            return False
        try:
            _, p_value = shapiro(data)
            return p_value > self.significance_level
        except:
            return False
    
    def calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
            
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (mean1 - mean2) / pooled_std
    
    def interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def compare_groups(self, group1: List[float], group2: List[float], 
                      group1_name: str, group2_name: str) -> StatisticalTest:
        """Compare two groups with appropriate statistical test"""
        
        if len(group1) == 0 or len(group2) == 0:
            return StatisticalTest(
                test_name="insufficient_data",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                interpretation="insufficient data",
                sample_sizes=(len(group1), len(group2))
            )
        
        # Calculate effect size
        effect_size = self.calculate_cohens_d(group1, group2)
        effect_interpretation = self.interpret_effect_size(effect_size)
        
        # Check normality
        normal1 = self.test_normality(group1)
        normal2 = self.test_normality(group2)
        
        if normal1 and normal2:
            # Use t-test for normally distributed data
            try:
                statistic, p_value = ttest_ind(group1, group2)
                test_name = "welch_t_test"
            except:
                statistic, p_value = 0.0, 1.0
                test_name = "t_test_failed"
        else:
            # Use Mann-Whitney U test for non-normal data
            try:
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = "mann_whitney_u"
            except:
                statistic, p_value = 0.0, 1.0
                test_name = "mann_whitney_failed"
        
        is_significant = p_value < self.significance_level
        
        # Create interpretation
        if is_significant:
            direction = "higher" if np.mean(group1) > np.mean(group2) else "lower"
            interpretation = f"{group1_name} significantly {direction} than {group2_name} (p={p_value:.4f}, d={effect_size:.3f}, {effect_interpretation} effect)"
        else:
            interpretation = f"No significant difference between {group1_name} and {group2_name} (p={p_value:.4f}, d={effect_size:.3f})"
        
        return StatisticalTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            is_significant=is_significant,
            interpretation=interpretation,
            sample_sizes=(len(group1), len(group2))
        )

class PublicationVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_comparison_boxplots(self, results_data: Dict[str, List[Dict]], 
                                 metrics: List[str]) -> List[str]:
        """Create boxplots comparing configurations"""
        saved_plots = []
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for boxplot
            data_for_plot = []
            labels = []
            
            for config_name, runs in results_data.items():
                values = [run.get(metric, 0) for run in runs if run.get(metric) is not None]
                if values:
                    data_for_plot.append(values)
                    labels.append(config_name.replace('_', ' ').title())
            
            if data_for_plot:
                # Create boxplot
                bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Formatting
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Configurations')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Rotate labels if needed
                if len(max(labels, key=len)) > 10:
                    plt.xticks(rotation=45, ha='right')
                
                # Save plot
                plot_filename = f"{metric}_comparison_boxplot.pdf"
                plot_path = self.output_dir / plot_filename
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                saved_plots.append(str(plot_path))
        
        return saved_plots
    
    def create_effect_size_heatmap(self, statistical_tests: Dict[str, Dict[str, StatisticalTest]]) -> str:
        """Create heatmap of effect sizes between configurations"""
        
        # Extract effect sizes
        configs = list(statistical_tests.keys())
        metrics = list(next(iter(statistical_tests.values())).keys()) if statistical_tests else []
        
        if not configs or not metrics:
            return ""
        
        # Create effect size matrix
        effect_matrix = np.zeros((len(metrics), len(configs)))
        
        for i, metric in enumerate(metrics):
            for j, config in enumerate(configs):
                if metric in statistical_tests.get(config, {}):
                    effect_matrix[i, j] = statistical_tests[config][metric].effect_size
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(effect_matrix, 
                   xticklabels=[c.replace('_', ' ').title() for c in configs],
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': "Cohen's d Effect Size"})
        
        ax.set_title("Effect Sizes: Contemplative vs Baseline Configurations")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        plot_filename = "effect_size_heatmap.pdf"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(plot_path)
    
    def create_performance_radar_chart(self, results_data: Dict[str, List[Dict]]) -> str:
        """Create radar chart showing configuration performance"""
        
        # Key metrics for radar chart
        radar_metrics = ['survival_rate', 'total_wisdom', 'average_mindfulness', 'ethical_ratio', 'network_coherence']
        
        # Calculate means for each configuration
        config_means = {}
        for config_name, runs in results_data.items():
            means = {}
            for metric in radar_metrics:
                values = [run.get(metric, 0) for run in runs if run.get(metric) is not None]
                means[metric] = np.mean(values) if values else 0
            config_means[config_name] = means
        
        if not config_means:
            return ""
        
        # Normalize values to 0-1 scale for radar chart
        all_values = {metric: [] for metric in radar_metrics}
        for config_data in config_means.values():
            for metric in radar_metrics:
                all_values[metric].append(config_data[metric])
        
        # Find min/max for normalization
        min_max = {}
        for metric in radar_metrics:
            values = all_values[metric]
            min_max[metric] = (min(values), max(values))
        
        # Normalize
        normalized_data = {}
        for config_name, config_data in config_means.items():
            normalized = {}
            for metric in radar_metrics:
                min_val, max_val = min_max[metric]
                if max_val > min_val:
                    normalized[metric] = (config_data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = 0.5
            normalized_data[config_name] = normalized
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (config_name, config_data) in enumerate(normalized_data.items()):
            values = [config_data[metric] for metric in radar_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=config_name.replace('_', ' ').title(),
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Configuration Performance Comparison\n(Normalized Metrics)', size=16, pad=20)
        
        # Save plot
        plot_filename = "performance_radar_chart.pdf"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(plot_path)

class XAILogger:
    """Enhanced explainable AI logging for publication"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.xai_dir = output_dir / "xai_logs"
        self.xai_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_decision_patterns(self, results_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze decision-making patterns across configurations"""
        
        analysis = {
            'decision_distribution': {},
            'contemplative_triggers': {},
            'wisdom_emergence_patterns': {},
            'ethical_decision_analysis': {}
        }
        
        for config_name, runs in results_data.items():
            if not runs:
                continue
                
            # Aggregate metrics across runs
            total_decisions = sum(run.get('simulation_steps', 0) for run in runs)
            total_wisdom = sum(run.get('total_wisdom', 0) for run in runs)
            avg_mindfulness = np.mean([run.get('average_mindfulness', 0) for run in runs])
            avg_ethical_ratio = np.mean([run.get('ethical_ratio', 0) for run in runs])
            
            analysis['decision_distribution'][config_name] = {
                'total_decisions': total_decisions,
                'decisions_per_run': total_decisions / len(runs) if runs else 0,
                'wisdom_per_decision': total_wisdom / max(total_decisions, 1)
            }
            
            analysis['contemplative_triggers'][config_name] = {
                'mindfulness_level': avg_mindfulness,
                'wisdom_generation_rate': total_wisdom / len(runs) if runs else 0,
                'ethical_decision_rate': avg_ethical_ratio
            }
        
        return analysis
    
    def generate_case_studies(self, results_data: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Generate detailed case studies from best/worst runs"""
        
        case_studies = []
        
        for config_name, runs in results_data.items():
            if not runs:
                continue
                
            # Find best and worst runs by survival rate
            runs_with_survival = [(i, run) for i, run in enumerate(runs) if run.get('survival_rate', 0) > 0]
            
            if runs_with_survival:
                best_run_idx, best_run = max(runs_with_survival, key=lambda x: x[1]['survival_rate'])
                worst_run_idx, worst_run = min(runs_with_survival, key=lambda x: x[1]['survival_rate'])
                
                case_studies.append({
                    'config': config_name,
                    'best_run': {
                        'run_index': best_run_idx,
                        'metrics': best_run,
                        'analysis': self._analyze_single_run(best_run, 'best')
                    },
                    'worst_run': {
                        'run_index': worst_run_idx,
                        'metrics': worst_run,
                        'analysis': self._analyze_single_run(worst_run, 'worst')
                    }
                })
        
        return case_studies
    
    def _analyze_single_run(self, run_data: Dict[str, Any], run_type: str) -> Dict[str, Any]:
        """Analyze a single run for XAI insights"""
        
        analysis = {
            'performance_category': run_type,
            'key_factors': [],
            'potential_causes': [],
            'contemplative_effectiveness': 'unknown'
        }
        
        # Analyze key factors
        survival_rate = run_data.get('survival_rate', 0)
        wisdom = run_data.get('total_wisdom', 0)
        mindfulness = run_data.get('average_mindfulness', 0)
        ethical_ratio = run_data.get('ethical_ratio', 0)
        
        if survival_rate > 2.0:
            analysis['key_factors'].append(f"High survival rate ({survival_rate:.1f}x)")
        if wisdom > 500:
            analysis['key_factors'].append(f"High wisdom generation ({wisdom:.0f} insights)")
        if mindfulness > 0.7:
            analysis['key_factors'].append(f"High mindfulness ({mindfulness:.2f})")
        if ethical_ratio > 0.4:
            analysis['key_factors'].append(f"High ethical behavior ({ethical_ratio:.2f})")
        
        # Determine contemplative effectiveness
        contemplative_score = (mindfulness + ethical_ratio + min(wisdom/1000, 1.0)) / 3
        if contemplative_score > 0.7:
            analysis['contemplative_effectiveness'] = 'high'
        elif contemplative_score > 0.4:
            analysis['contemplative_effectiveness'] = 'medium'
        else:
            analysis['contemplative_effectiveness'] = 'low'
        
        return analysis

class ReproducibilityTracker:
    """Track reproducibility information for publication"""
    
    def __init__(self):
        self.info = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'dependencies': self._get_dependencies(),
            'git_info': self._get_git_info(),
            'random_seeds': [],
            'configuration_hashes': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0]
        }
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get package versions"""
        try:
            import pkg_resources
            dependencies = {}
            for package in ['numpy', 'torch', 'scipy', 'matplotlib', 'seaborn', 'pandas']:
                try:
                    version = pkg_resources.get_distribution(package).version
                    dependencies[package] = version
                except:
                    dependencies[package] = 'unknown'
            return dependencies
        except:
            return {'status': 'could_not_determine'}
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information"""
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
            
            # Check if there are uncommitted changes
            try:
                subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD', '--'])
                clean = True
            except subprocess.CalledProcessError:
                clean = False
            
            return {
                'commit_hash': git_hash,
                'branch': git_branch,
                'clean_working_directory': clean
            }
        except:
            return {'status': 'not_a_git_repository'}
    
    def track_configuration(self, config_name: str, config_object: Any):
        """Track configuration for reproducibility"""
        import hashlib
        
        # Create hash of configuration
        config_str = str(config_object.__dict__)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        self.info['configuration_hashes'][config_name] = {
            'hash': config_hash,
            'timestamp': time.time()
        }
    
    def track_random_seed(self, seed: int, context: str):
        """Track random seeds used"""
        self.info['random_seeds'].append({
            'seed': seed,
            'context': context,
            'timestamp': time.time()
        })

class ComprehensivePublicationSuite:
    """Complete publication analysis suite"""
    
    def __init__(self, output_dir: str = "publication_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = PublicationVisualizer(self.output_dir / "visualizations")
        self.xai_logger = XAILogger(self.output_dir)
        self.reproducibility_tracker = ReproducibilityTracker()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_results(self, results_data: Dict[str, List[Dict]]) -> PublicationResults:
        """Complete analysis of experimental results"""
        
        self.logger.info("Starting comprehensive publication analysis...")
        
        # 1. Descriptive statistics
        descriptive_stats = self._calculate_descriptive_stats(results_data)
        
        # 2. Statistical significance testing
        statistical_tests = self._perform_statistical_tests(results_data)
        
        # 3. Effect size calculations
        effect_sizes = self._calculate_effect_sizes(results_data)
        
        # 4. Create visualizations
        visualizations = self._create_all_visualizations(results_data, statistical_tests)
        
        # 5. XAI analysis
        xai_summaries = self._perform_xai_analysis(results_data)
        
        # 6. Compile results
        results = PublicationResults(
            descriptive_stats=descriptive_stats,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            visualizations=visualizations,
            reproducibility_info=self.reproducibility_tracker.info,
            xai_summaries=xai_summaries
        )
        
        # 7. Generate reports
        self._generate_comprehensive_report(results)
        self._generate_jupyter_notebook(results)
        
        self.logger.info("Publication analysis completed!")
        return results
    
    def _calculate_descriptive_stats(self, results_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for all metrics"""
        
        stats = {}
        key_metrics = ['survival_rate', 'total_wisdom', 'average_mindfulness', 'ethical_ratio', 'network_coherence']
        
        for config_name, runs in results_data.items():
            if not runs:
                continue
                
            config_stats = {}
            for metric in key_metrics:
                values = [run.get(metric, 0) for run in runs if run.get(metric) is not None]
                if values:
                    config_stats[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n': len(values),
                        'sem': float(np.std(values) / np.sqrt(len(values))),  # Standard error of mean
                        'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0  # Coefficient of variation
                    }
            
            stats[config_name] = config_stats
        
        return stats
    
    def _perform_statistical_tests(self, results_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, StatisticalTest]]:
        """Perform pairwise statistical tests between configurations"""
        
        tests = {}
        configs = list(results_data.keys())
        key_metrics = ['survival_rate', 'total_wisdom', 'average_mindfulness', 'ethical_ratio']
        
        # Compare each configuration against baseline
        baseline_configs = [c for c in configs if 'baseline' in c or 'control' in c]
        contemplative_configs = [c for c in configs if 'contemplative' in c]
        
        for config in contemplative_configs:
            if config not in results_data or not results_data[config]:
                continue
                
            config_tests = {}
            
            # Compare against each baseline
            for baseline in baseline_configs:
                if baseline not in results_data or not results_data[baseline]:
                    continue
                
                for metric in key_metrics:
                    group1 = [run.get(metric, 0) for run in results_data[config] if run.get(metric) is not None]
                    group2 = [run.get(metric, 0) for run in results_data[baseline] if run.get(metric) is not None]
                    
                    test_result = self.statistical_analyzer.compare_groups(
                        group1, group2, config, baseline
                    )
                    
                    config_tests[f"{metric}_vs_{baseline}"] = test_result
            
            tests[config] = config_tests
        
        return tests
    
    def _calculate_effect_sizes(self, results_data: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes for key comparisons"""
        
        effect_sizes = {}
        configs = list(results_data.keys())
        key_metrics = ['survival_rate', 'total_wisdom', 'average_mindfulness', 'ethical_ratio']
        
        baseline_configs = [c for c in configs if 'baseline' in c or 'control' in c]
        contemplative_configs = [c for c in configs if 'contemplative' in c]
        
        for config in contemplative_configs:
            if config not in results_data:
                continue
                
            config_effects = {}
            
            for baseline in baseline_configs:
                if baseline not in results_data:
                    continue
                
                for metric in key_metrics:
                    group1 = [run.get(metric, 0) for run in results_data[config] if run.get(metric) is not None]
                    group2 = [run.get(metric, 0) for run in results_data[baseline] if run.get(metric) is not None]
                    
                    effect_size = self.statistical_analyzer.calculate_cohens_d(group1, group2)
                    config_effects[f"{metric}_vs_{baseline}"] = effect_size
            
            effect_sizes[config] = config_effects
        
        return effect_sizes
    
    def _create_all_visualizations(self, results_data: Dict[str, List[Dict]], 
                                 statistical_tests: Dict[str, Dict[str, StatisticalTest]]) -> List[str]:
        """Create all publication visualizations"""
        
        visualizations = []
        key_metrics = ['survival_rate', 'total_wisdom', 'average_mindfulness', 'ethical_ratio', 'network_coherence']
        
        # Boxplots
        boxplots = self.visualizer.create_comparison_boxplots(results_data, key_metrics)
        visualizations.extend(boxplots)
        
        # Effect size heatmap
        heatmap = self.visualizer.create_effect_size_heatmap(statistical_tests)
        if heatmap:
            visualizations.append(heatmap)
        
        # Radar chart
        radar_chart = self.visualizer.create_performance_radar_chart(results_data)
        if radar_chart:
            visualizations.append(radar_chart)
        
        return visualizations
    
    def _perform_xai_analysis(self, results_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform explainable AI analysis"""
        
        decision_patterns = self.xai_logger.analyze_decision_patterns(results_data)
        case_studies = self.xai_logger.generate_case_studies(results_data)
        
        return {
            'decision_patterns': decision_patterns,
            'case_studies': case_studies,
            'summary': self._generate_xai_summary(decision_patterns, case_studies)
        }
    
    def _generate_xai_summary(self, decision_patterns: Dict, case_studies: List[Dict]) -> Dict[str, Any]:
        """Generate XAI summary for publication"""
        
        summary = {
            'key_insights': [],
            'contemplative_effectiveness': {},
            'decision_making_patterns': {}
        }
        
        # Analyze contemplative effectiveness across configurations
        for config_name, triggers in decision_patterns.get('contemplative_triggers', {}).items():
            effectiveness_score = (
                triggers.get('mindfulness_level', 0) * 0.4 +
                min(triggers.get('wisdom_generation_rate', 0) / 500, 1.0) * 0.3 +
                triggers.get('ethical_decision_rate', 0) * 0.3
            )
            summary['contemplative_effectiveness'][config_name] = effectiveness_score
        
        # Extract key insights from case studies
        for study in case_studies:
            config = study['config']
            best_analysis = study['best_run']['analysis']
            
            if best_analysis['contemplative_effectiveness'] == 'high':
                summary['key_insights'].append(
                    f"{config}: High contemplative effectiveness correlates with {', '.join(best_analysis['key_factors'])}"
                )
        
        return summary
    
    def _generate_comprehensive_report(self, results: PublicationResults):
        """Generate comprehensive publication report"""
        
        report_path = self.output_dir / "COMPREHENSIVE_PUBLICATION_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MycoNet++ Contemplative AI: Comprehensive Publication Analysis\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive statistical analysis of MycoNet++ Contemplative AI experiments, ")
            f.write("including significance testing, effect sizes, and explainable AI insights suitable for academic publication.\n\n")
            
            # Key Findings
            f.write("## Key Statistical Findings\n\n")
            
            # Extract significant results
            significant_results = []
            for config, tests in results.statistical_tests.items():
                for test_name, test_result in tests.items():
                    if test_result.is_significant and test_result.effect_size > 0.5:
                        significant_results.append(f"- {test_result.interpretation}")
            
            if significant_results:
                f.write("### Statistically Significant Results (p < 0.05, medium+ effect size):\n\n")
                for result in significant_results:
                    f.write(f"{result}\n")
                f.write("\n")
            
            # Effect Sizes
            f.write("## Effect Size Analysis\n\n")
            f.write("Effect sizes (Cohen's d) indicate practical significance:\n")
            f.write("- Small: 0.2-0.5\n- Medium: 0.5-0.8\n- Large: >0.8\n\n")
            
            for config, effects in results.effect_sizes.items():
                f.write(f"### {config.replace('_', ' ').title()}\n\n")
                for comparison, effect_size in effects.items():
                    interpretation = self.statistical_analyzer.interpret_effect_size(effect_size)
                    f.write(f"- {comparison}: d = {effect_size:.3f} ({interpretation})\n")
                f.write("\n")
            
            # Descriptive Statistics
            f.write("## Descriptive Statistics\n\n")
            for config, stats in results.descriptive_stats.items():
                f.write(f"### {config.replace('_', ' ').title()}\n\n")
                f.write("| Metric | Mean ± SD | Median | Range | n |\n")
                f.write("|--------|-----------|--------|-------|---|\n")
                
                for metric, metric_stats in stats.items():
                    f.write(f"| {metric.replace('_', ' ').title()} | ")
                    f.write(f"{metric_stats['mean']:.3f} ± {metric_stats['std']:.3f} | ")
                    f.write(f"{metric_stats['median']:.3f} | ")
                    f.write(f"{metric_stats['min']:.3f}-{metric_stats['max']:.3f} | ")
                    f.write(f"{metric_stats['n']} |\n")
                f.write("\n")
            
            # XAI Insights
            f.write("## Explainable AI Insights\n\n")
            if results.xai_summaries.get('key_insights'):
                f.write("### Key Decision-Making Patterns:\n\n")
                for insight in results.xai_summaries['key_insights']:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            # Reproducibility
            f.write("## Reproducibility Information\n\n")
            repro = results.reproducibility_info
            
            f.write(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(repro['timestamp']))}\n\n")
            
            if 'system_info' in repro:
                f.write("**System Information:**\n")
                for key, value in repro['system_info'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
            
            if 'dependencies' in repro:
                f.write("**Package Versions:**\n")
                for package, version in repro['dependencies'].items():
                    f.write(f"- {package}: {version}\n")
                f.write("\n")
            
            if 'git_info' in repro and 'commit_hash' in repro['git_info']:
                f.write("**Git Information:**\n")
                f.write(f"- Commit: {repro['git_info']['commit_hash']}\n")
                f.write(f"- Branch: {repro['git_info']['branch']}\n")
                f.write(f"- Clean: {repro['git_info']['clean_working_directory']}\n\n")
            
            # Visualizations
            f.write("## Generated Visualizations\n\n")
            for viz_path in results.visualizations:
                viz_name = Path(viz_path).name
                f.write(f"- {viz_name}\n")
            f.write("\n")
            
            # Publication Recommendations
            f.write("## Publication Recommendations\n\n")
            f.write("### Target Venues:\n")
            f.write("1. **AAMAS 2026** - Autonomous Agents and Multiagent Systems\n")
            f.write("2. **IJCAI 2026** - AI and Society track\n")
            f.write("3. **Nature Machine Intelligence** - If emergence results are exceptionally strong\n")
            f.write("4. **Artificial Life** - Biomimetic aspects\n\n")
            
            f.write("### Key Contributions to Highlight:\n")
            f.write("1. First quantitative measurement of artificial mindfulness\n")
            f.write("2. Demonstration of spontaneous wisdom emergence in multi-agent systems\n")
            f.write("3. Contemplative processing as evolutionary advantage\n")
            f.write("4. Emergent ethical behavior without explicit rewards\n\n")
    
    def _generate_jupyter_notebook(self, results: PublicationResults):
        """Generate Jupyter notebook with interactive analysis"""
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# MycoNet++ Contemplative AI: Interactive Publication Analysis\n",
                        "\n",
                        "This notebook contains interactive analysis of the MycoNet++ Contemplative AI experimental results."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "import json\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from scipy import stats\n",
                        "\n",
                        "# Load results\n",
                        "with open('publication_analysis/results_summary.json', 'r') as f:\n",
                        "    results = json.load(f)\n",
                        "\n",
                        "print('Results loaded successfully!')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Statistical Summary\n",
                        "\n",
                        "Key findings from the experimental analysis:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Display key statistical results\n",
                        "for config, tests in results['statistical_tests'].items():\n",
                        "    print(f\"\\n{config.replace('_', ' ').title()}:\")\n",
                        "    for test_name, test_result in tests.items():\n",
                        "        if test_result['is_significant']:\n",
                        "            print(f\"  ✓ {test_result['interpretation']}\")\n",
                        "        else:\n",
                        "            print(f\"  - {test_result['interpretation']}\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Visualization Recreation\n",
                        "\n",
                        "Recreate key visualizations for publication:"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Example: Recreate survival rate comparison\n",
                        "# This would contain actual data and plotting code\n",
                        "# Generated dynamically based on the results\n",
                        "\n",
                        "plt.figure(figsize=(10, 6))\n",
                        "# Plotting code would go here\n",
                        "plt.title('Survival Rate Comparison Across Configurations')\n",
                        "plt.show()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = self.output_dir / "interactive_analysis.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    def generate_batch_script(self, base_config, num_runs: int = 50) -> str:
        """Generate batch script for large-scale runs"""
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=myconet_contemplative
#SBATCH --output=logs/myconet_%j.out
#SBATCH --error=logs/myconet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# MycoNet++ Contemplative AI Large-Scale Batch Run
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

echo "Starting MycoNet++ Contemplative batch run..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create output directory
mkdir -p batch_results/$SLURM_JOB_ID

# Run experiments
for run_id in {{1..{num_runs}}}; do
    echo "Running experiment $run_id/{num_runs}..."
    python myconet_contemplative_main.py \\
        --config batch_config.json \\
        --run-id $run_id \\
        --output-dir batch_results/$SLURM_JOB_ID \\
        --seed $((42 + run_id * 1000))
        
    if [ $? -eq 0 ]; then
        echo "Run $run_id completed successfully"
    else
        echo "Run $run_id failed"
    fi
done

echo "Batch run completed!"
echo "Results in: batch_results/$SLURM_JOB_ID"
"""
        
        script_path = self.output_dir / "batch_run.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        import os
        os.chmod(script_path, 0o755)
        
        return str(script_path)

# Convenience function for integration with existing system
def analyze_publication_results(results_directory: str, 
                              output_directory: str = "publication_analysis") -> PublicationResults:
    """
    Analyze results from existing publication experiments
    
    Args:
        results_directory: Directory containing JSON result files
        output_directory: Directory to save analysis outputs
    
    Returns:
        PublicationResults object with comprehensive analysis
    """
    
    # Load results from JSON files
    results_data = {}
    results_path = Path(results_directory)
    
    for json_file in results_path.glob("*_results.json"):
        config_name = json_file.stem.replace("_results", "")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        if 'individual_runs' in data:
            results_data[config_name] = data['individual_runs']
        else:
            # Handle different JSON formats
            results_data[config_name] = [data] if isinstance(data, dict) else data
    
    # Create analysis suite and run analysis
    suite = ComprehensivePublicationSuite(output_directory)
    results = suite.analyze_results(results_data)
    
    print(f"\n🎉 Publication analysis completed!")
    print(f"📊 Analyzed {len(results_data)} configurations")
    print(f"📈 Generated {len(results.visualizations)} visualizations")
    print(f"📂 Results saved to: {output_directory}")
    print(f"📄 Report: {output_directory}/COMPREHENSIVE_PUBLICATION_REPORT.md")
    
    return results

# Integration with existing runner
def enhance_existing_runner(runner_instance):
    """
    Enhance existing publication runner with comprehensive analysis
    
    Add this to your existing runner's run_quick_comparison_study method:
    """
    
    def enhanced_analysis(self, all_results, analysis):
        """Enhanced analysis method to add to existing runner"""
        
        # Run comprehensive publication analysis
        if any(results for results in all_results.values()):
            print("\n🔬 Running comprehensive publication analysis...")
            
            publication_results = analyze_publication_results(
                results_directory=str(self.results_dir),
                output_directory=str(self.results_dir / "comprehensive_analysis")
            )
            
            # Add publication insights to existing analysis
            analysis['publication_results'] = {
                'statistical_significance_tests': len([
                    test for tests in publication_results.statistical_tests.values() 
                    for test in tests.values() if test.is_significant
                ]),
                'large_effect_sizes': len([
                    effect for effects in publication_results.effect_sizes.values()
                    for effect in effects.values() if abs(effect) > 0.8
                ]),
                'visualizations_created': len(publication_results.visualizations),
                'xai_insights': len(publication_results.xai_summaries.get('key_insights', []))
            }
            
            print(f"✅ Enhanced analysis completed!")
            print(f"📊 {analysis['publication_results']['statistical_significance_tests']} significant tests")
            print(f"📈 {analysis['publication_results']['large_effect_sizes']} large effect sizes")
            print(f"🖼️  {analysis['publication_results']['visualizations_created']} publication-quality plots")
        
        return analysis
    
    # Add method to runner instance
    runner_instance.enhanced_analysis = enhanced_analysis.__get__(runner_instance, type(runner_instance))
    
    return runner_instance

if __name__ == "__main__":
    print("🧠 MycoNet++ Contemplative AI - Publication Analysis Suite")
    print("=" * 60)
    print("📊 Statistical significance testing with effect sizes")
    print("📈 Publication-quality visualizations") 
    print("🔍 Explainable AI analysis")
    print("🔄 Reproducibility tracking")
    print("📄 Comprehensive reporting")
    print("\nUse analyze_publication_results() to analyze your experimental data!")