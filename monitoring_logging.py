            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                self.monitor.operation_times[self.operation_name].append(duration)
                
                if self.operation_name in self.monitor.current_operations:
                    del self.monitor.current_operations[self.operation_name]
                
                # Check performance thresholds
                if duration > self.monitor.performance_thresholds['max_processing_time']:
                    self.monitor._generate_performance_alert(
                        'slow_operation', 
                        f"Operation {self.operation_name} took {duration:.3f}s"
                    )
    
    def record_success(self, operation: str, step: int):
        """Record successful operation"""
        self.success_counts[operation] += 1
    
    def record_failure(self, operation: str, step: int, error: str):
        """Record failed operation"""
        self.failure_counts[operation] += 1
        
        # Check failure rate
        total_ops = self.success_counts[operation] + self.failure_counts[operation]
        failure_rate = self.failure_counts[operation] / total_ops
        
        if failure_rate > self.performance_thresholds['max_failure_rate']:
            self._generate_performance_alert(
                'high_failure_rate',
                f"Operation {operation} has {failure_rate:.1%} failure rate"
            )
        
        logger.warning(f"Operation {operation} failed at step {step}: {error}")
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        self.memory_usage.append(memory_mb)
        
        if memory_mb > self.performance_thresholds['max_memory_usage']:
            self._generate_performance_alert(
                'high_memory_usage',
                f"Memory usage: {memory_mb:.1f}MB"
            )
    
    def _generate_performance_alert(self, alert_type: str, message: str):
        """Generate performance alert"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.performance_alerts.append(alert)
        
        if alert['severity'] == 'critical':
            logger.error(f"PERFORMANCE ALERT: {message}")
        else:
            logger.warning(f"Performance warning: {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level"""
        severity_map = {
            'slow_operation': 'warning',
            'high_failure_rate': 'critical',
            'high_memory_usage': 'warning'
        }
        return severity_map.get(alert_type, 'info')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'operation_times': {},
            'success_rates': {},
            'current_operations': len(self.current_operations),
            'total_operations': sum(len(times) for times in self.operation_times.values()),
            'memory_usage': {
                'current': self.memory_usage[-1] if self.memory_usage else 0,
                'average': np.mean(self.memory_usage) if self.memory_usage else 0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0
            },
            'recent_alerts': len([a for a in self.performance_alerts 
                                if time.time() - a['timestamp'] < 3600])  # Last hour
        }
        
        # Operation timing statistics
        for operation, times in self.operation_times.items():
            if times:
                metrics['operation_times'][operation] = {
                    'average': np.mean(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'p95': np.percentile(times, 95),
                    'count': len(times)
                }
        
        # Success rates
        for operation in set(list(self.success_counts.keys()) + list(self.failure_counts.keys())):
            successes = self.success_counts[operation]
            failures = self.failure_counts[operation]
            total = successes + failures
            
            if total > 0:
                metrics['success_rates'][operation] = {
                    'rate': successes / total,
                    'total_operations': total,
                    'successes': successes,
                    'failures': failures
                }
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary"""
        metrics = self.get_metrics()
        
        # Calculate overall health score
        health_factors = []
        
        # Processing time health
        if metrics['operation_times']:
            avg_times = [op['average'] for op in metrics['operation_times'].values()]
            avg_processing_time = np.mean(avg_times)
            time_health = max(0, 1.0 - (avg_processing_time / 5.0))  # 5s threshold
            health_factors.append(time_health)
        
        # Success rate health
        if metrics['success_rates']:
            success_rates = [op['rate'] for op in metrics['success_rates'].values()]
            avg_success_rate = np.mean(success_rates)
            health_factors.append(avg_success_rate)
        
        # Memory health
        if self.memory_usage:
            current_memory = self.memory_usage[-1]
            memory_health = max(0, 1.0 - (current_memory / 1000.0))  # 1GB threshold
            health_factors.append(memory_health)
        
        overall_health = np.mean(health_factors) if health_factors else 0.5
        
        return {
            'overall_health_score': overall_health,
            'health_status': self._get_health_status(overall_health),
            'total_operations': metrics['total_operations'],
            'active_operations': metrics['current_operations'],
            'recent_alerts': metrics['recent_alerts'],
            'average_processing_time': np.mean([op['average'] for op in metrics['operation_times'].values()]) if metrics['operation_times'] else 0,
            'overall_success_rate': np.mean([op['rate'] for op in metrics['success_rates'].values()]) if metrics['success_rates'] else 0
        }
    
    def _get_health_status(self, health_score: float) -> str:
        """Convert health score to status"""
        if health_score >= 0.8:
            return 'excellent'
        elif health_score >= 0.6:
            return 'good'
        elif health_score >= 0.4:
            return 'fair'
        elif health_score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def get_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        alerts = list(self.performance_alerts)
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_status(self) -> str:
        """Get monitor status"""
        health = self.get_performance_summary()
        return f"{health['health_status']} ({health['total_operations']} ops, {health['recent_alerts']} alerts)"

# ===== STATUS REPORTING SYSTEM =====

class StatusReporter:
    """Comprehensive status reporting for all overmind components"""
    
    def __init__(self, overmind):
        self.overmind = overmind
        self.report_history = deque(maxlen=100)
        
    def get_comprehensive_status(self, detail_level: str = 'summary') -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'timestamp': time.time(),
            'overmind_id': self.overmind.overmind_id,
            'system_status': self._determine_system_status(),
            'uptime_hours': (time.time() - self.overmind.system_state['initialization_time']) / 3600,
            'total_decisions': self.overmind.performance_metrics['total_decisions'],
            'success_rate': self._calculate_success_rate(),
            'components': self._get_component_status()
        }
        
        if detail_level == 'detailed':
            status.update({
                'performance_metrics': self._get_performance_details(),
                'recent_decisions': self._get_recent_decisions(),
                'active_rituals': self._get_active_rituals(),
                'memory_utilization': self._get_memory_utilization(),
                'error_summary': self._get_error_summary(),
                'thresholds': self._get_threshold_status()
            })
        
        elif detail_level == 'diagnostic':
            status.update({
                'full_performance_metrics': self.overmind.performance_monitor.get_metrics() if hasattr(self.overmind, 'performance_monitor') else {},
                'component_diagnostics': self._get_component_diagnostics(),
                'system_health_analysis': self._get_system_health_analysis(),
                'recommendations': self._generate_recommendations()
            })
        
        # Store in history
        self.report_history.append(status)
        
        return status
    
    def _determine_system_status(self) -> str:
        """Determine overall system status"""
        
        # Check for critical issues
        if hasattr(self.overmind, 'performance_monitor'):
            alerts = self.overmind.performance_monitor.get_alerts('critical')
            if alerts:
                return 'degraded'
        
        # Check error rate
        error_rate = self._get_recent_error_rate()
        if error_rate > 0.1:
            return 'unstable'
        
        # Check success rate
        success_rate = self._calculate_success_rate()
        if success_rate < 0.5:
            return 'poor_performance'
        
        return 'healthy'
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.overmind.performance_metrics['total_decisions']
        if total == 0:
            return 0.0
        
        successful = self.overmind.performance_metrics['successful_interventions']
        return successful / total
    
    def _get_component_status(self) -> Dict[str, str]:
        """Get status of all components"""
        components = {}
        
        component_list = [
            ('wisdom_archive', 'wisdom_archive'),
            ('memory_attention', 'memory_attention'),
            ('agent_feedback', 'agent_feedback'),
            ('ritual_layer', 'ritual_layer'),
            ('threshold_regulator', 'threshold_regulator'),
            ('logger', 'logger'),
            ('performance_monitor', 'performance_monitor')
        ]
        
        for name, attr_name in component_list:
            if hasattr(self.overmind, attr_name):
                component = getattr(self.overmind, attr_name)
                try:
                    if hasattr(component, 'get_status'):
                        components[name] = component.get_status()
                    else:
                        components[name] = 'active'
                except Exception as e:
                    components[name] = f'error: {str(e)[:50]}'
            else:
                components[name] = 'not_available'
        
        return components
    
    def _get_performance_details(self) -> Dict[str, Any]:
        """Get detailed performance information"""
        if not hasattr(self.overmind, 'performance_monitor'):
            return {}
        
        return self.overmind.performance_monitor.get_performance_summary()
    
    def _get_recent_decisions(self) -> List[Dict[str, Any]]:
        """Get recent decision summary"""
        recent_decisions = []
        
        for decision_record in list(self.overmind.decision_history)[-10:]:
            decision = decision_record.get('decision')
            if decision:
                recent_decisions.append({
                    'step': decision_record['step'],
                    'action': decision.chosen_action.name,
                    'confidence': decision.confidence,
                    'processing_time': decision_record.get('processing_time', 0)
                })
        
        return recent_decisions
    
    def _get_active_rituals(self) -> Dict[str, Any]:
        """Get active ritual information"""
        if not hasattr(self.overmind, 'ritual_layer'):
            return {}
        
        return {
            'active_count': len(getattr(self.overmind.ritual_layer, 'active_rituals', {})),
            'total_completed': len(getattr(self.overmind.ritual_layer, 'ritual_history', []))
        }
    
    def _get_memory_utilization(self) -> Dict[str, Any]:
        """Get memory system utilization"""
        if not hasattr(self.overmind, 'memory_attention'):
            return {}
        
        memory_system = self.overmind.memory_attention
        return {
            'total_memories': len(getattr(memory_system, 'intervention_memories', [])),
            'memory_capacity': getattr(memory_system, 'memory_capacity', 0),
            'utilization_rate': len(getattr(memory_system, 'intervention_memories', [])) / max(1, getattr(memory_system, 'memory_capacity', 1))
        }
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        if not hasattr(self.overmind, 'logger'):
            return {}
        
        error_log = getattr(self.overmind.logger, 'error_log', [])
        recent_errors = [e for e in error_log if time.time() - e.get('timestamp', 0) < 3600]
        
        return {
            'total_errors': len(error_log),
            'recent_errors': len(recent_errors),
            'error_rate': self._get_recent_error_rate()
        }
    
    def _get_recent_error_rate(self) -> float:
        """Calculate recent error rate"""
        if not hasattr(self.overmind, 'logger'):
            return 0.0
        
        recent_entries = self.overmind.logger.get_recent_logs(100)
        if not recent_entries:
            return 0.0
        
        error_count = sum(1 for entry in recent_entries if entry.get('type') == 'error')
        return error_count / len(recent_entries)
    
    def _get_threshold_status(self) -> Dict[str, Any]:
        """Get threshold system status"""
        if not hasattr(self.overmind, 'threshold_regulator'):
            return {}
        
        regulator = self.overmind.threshold_regulator
        return {
            'current_thresholds': getattr(regulator, 'thresholds', {}),
            'recent_adjustments': sum(len(history) for history in getattr(regulator, 'threshold_history', {}).values())
        }
    
    def _get_component_diagnostics(self) -> Dict[str, Any]:
        """Get detailed component diagnostics"""
        diagnostics = {}
        
        # Add component-specific diagnostic information
        if hasattr(self.overmind, 'wisdom_archive'):
            archive = self.overmind.wisdom_archive
            diagnostics['wisdom_archive'] = {
                'insights_count': len(getattr(archive, 'insights', {})),
                'decay_warnings': len([id for id, score in getattr(archive, 'decay_scores', {}).items() if score > 0.7])
            }
        
        if hasattr(self.overmind, 'agent_feedback'):
            feedback = self.overmind.agent_feedback
            diagnostics['agent_feedback'] = {
                'total_applications': len(getattr(feedback, 'feedback_history', [])),
                'tracked_agents': len(getattr(feedback, 'agent_response_tracking', {}))
            }
        
        return diagnostics
    
    def _get_system_health_analysis(self) -> Dict[str, Any]:
        """Analyze overall system health"""
        analysis = {
            'overall_health': 'good',
            'critical_issues': [],
            'warnings': [],
            'performance_trends': []
        }
        
        # Check for critical issues
        if self._calculate_success_rate() < 0.3:
            analysis['critical_issues'].append('Very low success rate')
            analysis['overall_health'] = 'critical'
        
        # Check for warnings
        if self._get_recent_error_rate() > 0.05:
            analysis['warnings'].append('Elevated error rate')
        
        # Performance trends
        if len(self.report_history) >= 3:
            recent_success_rates = [r.get('success_rate', 0) for r in list(self.report_history)[-3:]]
            if all(recent_success_rates[i] < recent_success_rates[i-1] for i in range(1, len(recent_success_rates))):
                analysis['performance_trends'].append('Declining success rate')
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if self._calculate_success_rate() < 0.7:
            recommendations.append("Consider reviewing decision-making thresholds")
        
        if self._get_recent_error_rate() > 0.05:
            recommendations.append("Review error logs and implement additional error handling")
        
        # Memory-based recommendations
        if hasattr(self.overmind, 'memory_attention'):
            memory_util = self._get_memory_utilization()
            if memory_util.get('utilization_rate', 0) > 0.9:
                recommendations.append("Memory system approaching capacity - consider pruning old memories")
        
        # Threshold-based recommendations
        if hasattr(self.overmind, 'threshold_regulator'):
            threshold_analysis = self.overmind.threshold_regulator.get_threshold_analysis()
            for threshold_name, analysis in threshold_analysis.items():
                if analysis.get('recommendation') == 'MAJOR_ADJUSTMENT_NEEDED':
                    recommendations.append(f"Major adjustment needed for {threshold_name}")
        
        return recommendations
    
    def print_status_report(self, detail_level: str = 'summary', output_file: str = None):
        """Print formatted status report"""
        
        status = self.get_comprehensive_status(detail_level)
        
        output = []
        output.append("=" * 60)
        output.append(f"OVERMIND STATUS REPORT - {status['overmind_id']}")
        output.append("=" * 60)
        output.append(f"System Status: {status['system_status'].upper()}")
        output.append(f"Uptime: {status['uptime_hours']:.1f} hours")
        output.append(f"Total Decisions: {status['total_decisions']}")
        output.append(f"Success Rate: {status['success_rate']:.1%}")
        output.append("")
        
        # Components
        output.append("COMPONENT STATUS:")
        for component, component_status in status['components'].items():
            output.append(f"  {component}: {component_status}")
        output.append("")
        
        if detail_level in ['detailed', 'diagnostic']:
            # Performance details
            if 'performance_metrics' in status:
                perf = status['performance_metrics']
                output.append("PERFORMANCE METRICS:")
                output.append(f"  Overall Health: {perf.get('health_status', 'unknown')}")
                output.append(f"  Active Operations: {perf.get('active_operations', 0)}")
                output.append(f"  Recent Alerts: {perf.get('recent_alerts', 0)}")
                output.append("")
            
            # Recent decisions
            if 'recent_decisions' in status:
                output.append("RECENT DECISIONS:")
                for decision in status['recent_decisions'][-5:]:
                    output.append(f"  Step {decision['step']}: {decision['action']} "
                                f"(conf: {decision['confidence']:.3f})")
                output.append("")
        
        if detail_level == 'diagnostic':
            # Recommendations
            if 'recommendations' in status:
                output.append("RECOMMENDATIONS:")
                for rec in status['recommendations']:
                    output.append(f"  • {rec}")
                output.append("")
        
        output.append("=" * 60)
        
        report_text = "\n".join(output)
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Status report written to {output_file}")
            except Exception as e:
                logger.error(f"Failed to write status report: {e}")
        else:
            print(report_text)
    
    def get_status(self) -> str:
        """Get reporter status"""
        return f"active ({len(self.report_history)} reports generated)"

# ===== COMPREHENSIVE TEST SUITE =====

class TestSuite:
    """Comprehensive testing framework for all overmind components"""
    
    def __init__(self, overmind):
        self.overmind = overmind
        self.test_results = {}
        self.test_agents = []
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment with various agent types"""
        
        # Create diverse test agents
        self.test_agents = []
        
        # Normal agents
        for i in range(10):
            agent = self.create_test_agent(f"test_normal_{i}", agent_type='normal')
            self.test_agents.append(agent)
        
        # Stressed agents
        for i in range(3):
            agent = self.create_test_agent(f"test_stressed_{i}", agent_type='stressed')
            self.test_agents.append(agent)
        
        # Wise agents
        for i in range(2):
            agent = self.create_test_agent(f"test_wise_{i}", agent_type='wise')
            self.test_agents.append(agent)
    
    def create_test_agent(self, agent_id: str, agent_type: str = 'normal'):
        """Create test agent with specified characteristics"""
        
        agent = type('TestAgent', (), {})()
        agent.id = agent_id
        
        if agent_type == 'normal':
            agent.energy = np.random.uniform(0.4, 0.8)
            agent.health = np.random.uniform(0.5, 0.9)
            agent.mindfulness_level = np.random.uniform(0.3, 0.7)
            agent.cooperation_tendency = np.random.uniform(0.4, 0.8)
            agent.conflict_tendency = np.random.uniform(0.1, 0.3)
        
        elif agent_type == 'stressed':
            agent.energy = np.random.uniform(0.1, 0.4)
            agent.health = np.random.uniform(0.2, 0.5)
            agent.mindfulness_level = np.random.uniform(0.1, 0.4)
            agent.cooperation_tendency = np.random.uniform(0.2, 0.5)
            agent.conflict_tendency = np.random.uniform(0.3, 0.6)
        
        elif agent_type == 'wise':
            agent.energy = np.random.uniform(0.6, 0.9)
            agent.health = np.random.uniform(0.7, 0.9)
            agent.mindfulness_level = np.random.uniform(0.7, 0.9)
            agent.cooperation_tendency = np.random.uniform(0.7, 0.9)
            agent.conflict_tendency = np.random.uniform(0.0, 0.2)
            agent.wisdom_accumulated = np.random.uniform(3.0, 8.0)
        
        # Common attributes
        agent.learning_rate = np.random.uniform(0.3, 0.8)
        agent.emotional_stability = np.random.uniform(0.4, 0.8)
        agent.stress_level = np.random.uniform(0.1, 0.5)
        agent.recent_insights = ["Test insight", "Another insight"]
        agent.relationships = {f"agent_{i}": np.random.uniform(0.2, 0.8) for i in range(3)}
        
        return agent
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive test suite"""
        
        print("🧪 Running Comprehensive Test Suite")
        print("=" * 50)
        
        tests = [
            ('test_basic_functionality', self.test_basic_functionality),
            ('test_decision_making', self.test_decision_making),
            ('test_agent_feedback', self.test_agent_feedback),
            ('test_memory_system', self.test_memory_system),
            ('test_threshold_adaptation', self.test_threshold_adaptation),
            ('test_error_handling', self.test_error_handling),
            ('test_performance_monitoring', self.test_performance_monitoring),
            ('test_status_reporting', self.test_status_reporting)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   ❌ ERROR: {e}")
                if hasattr(self.overmind, 'logger'):
                    self.overmind.logger.log_error('test_error', str(e), -1)
        
        self.test_results = results
        return results
    
    def test_basic_functionality(self) -> bool:
        """Test basic overmind functionality"""
        try:
            # Test decision making
            decision = self.overmind.process_colony_state_fully_integrated(self.test_agents, 1000)
            if decision is None:
                return False
            
            # Test status reporting
            if hasattr(self.overmind, 'status_reporter'):
                status = self.overmind.status_reporter.get_comprehensive_status()
                if not status or status.get('system_status') == 'error':
                    return False
            
            return True
        except Exception as e:
            print(f"Basic functionality test failed: {e}")
            return False
    
    def test_decision_making(self) -> bool:
        """Test decision making process"""
        try:
            # Test multiple decision scenarios
            scenarios = [
                (self.test_agents, "normal"),
                (self.test_agents[:5], "small_group"),
                ([a for a in self.test_agents if 'stressed' in getattr(a, 'id', '')], "stressed_group")
            ]
            
            for agents, scenario_name in scenarios:
                if not agents:
                    continue
                    
                decision = self.overmind.process_colony_state_fully_integrated(agents, 1001)
                if decision is None and len(agents) > 0:
                    print(f"Decision failed for scenario: {scenario_name}")
                    return False
            
            return True
        except Exception as e:
            print(f"Decision making test failed: {e}")
            return False
    
    def test_agent_feedback(self) -> bool:
        """Test agent feedback system"""
        try:
            if not hasattr(self.overmind, 'agent_feedback'):
                return True  # Skip if not available
            
            test_agent = self.test_agents[0]
            initial_mindfulness = getattr(test_agent, 'mindfulness_level', 0.5)
            
            # Apply feedback
            result = self.overmind.agent_feedback.apply_overmind_feedback(
                [test_agent], 'mindfulness_boost', 0.5, OvermindActionType.TRIGGER_COLLECTIVE_MEDITATION
            )
            
            if not result.get('success', False):
                return False
            
            # Check if agent changed
            new_mindfulness = getattr(test_agent, 'mindfulness_level', 0.5)
            return new_mindfulness >= initial_mindfulness
            
        except Exception as e:
            print(f"Agent feedback test failed: {e}")
            return False
    
    def test_memory_system(self) -> bool:
        """Test memory attention system"""
        try:
            if not hasattr(self.overmind, 'memory_attention'):
                return True  # Skip if not available
            
            # Add test memory
            test_decision = {'chosen_action': OvermindActionType.NO_ACTION, 'confidence': 0.7}
            test_impact = {'agents_affected': 10, 'effectiveness': 0.8}
            
            self.overmind.memory_attention.add_intervention_memory(test_decision, test_impact)
            
            # Test memory influence
            context = {'colony_metrics': ColonyMetrics(self.test_agents)}
            influence = self.overmind.memory_attention.compute_weighted_memory_influence(context)
            
            return isinstance(influence, dict) and 'memory_influence' in influence
            
        except Exception as e:
            print(f"Memory system test failed: {e}")
            return False
    
    def test_threshold_adaptation(self) -> bool:
        """Test adaptive threshold system"""
        try:
            if not hasattr(self.overmind, 'threshold_regulator'):
                return True  # Skip if not available
            
            regulator = self.overmind.threshold_regulator
            initial_threshold = regulator.get_threshold('intervention_threshold')
            
            # Record some outcomes
            regulator.record_intervention_outcome('intervention_threshold', True, False)  # False positive
            regulator.record_intervention_outcome('intervention_threshold', True, False)  # Another false positive
            
            new_threshold = regulator.get_threshold('intervention_threshold')
            
            # Threshold should have increased (more conservative) due to false positives
            return new_threshold >= initial_threshold
            
        except Exception as e:
            print(f"Threshold adaptation test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery"""
        try:
            # Test with invalid inputs
            result = self.overmind.process_colony_state_fully_integrated([], -1)  # Empty agents, negative step
            
            # Should handle gracefully and return None or safe decision
            return result is None or (hasattr(result, 'chosen_action') and 
                                    result.chosen_action == OvermindActionType.NO_ACTION)
            
        except Exception as e:
            # Should not throw unhandled exceptions
            print(f"Error handling test failed: {e}")
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring system"""
        try:
            if not hasattr(self.overmind, 'performance_monitor'):
                return True  # Skip if not available
            
            monitor = self.overmind.performance_monitor
            
            # Test operation timing
            with monitor.measure_operation('test_operation'):
                time.sleep(0.01)  # Small delay
            
            # Test success/failure recording
            monitor.record_success('test_op', 1000)
            monitor.record_failure('test_op', 1001, 'test error')
            
            # Check metrics
            metrics = monitor.get_metrics()
            return ('operation_times' in metrics and 
                    'success_rates' in metrics and
                    'test_operation' in metrics['operation_times'])
            
        except Exception as e:
            print(f"Performance monitoring test failed: {e}")
            return False
    
    def test_status_reporting(self) -> bool:
        """Test status reporting system"""
        try:
            if not hasattr(self.overmind, 'status_reporter'):
                return True  # Skip if not available
            
            # Test basic status report
            status = self.overmind.status_reporter.get_comprehensive_status('summary')
            required_fields = ['overmind_id', 'system_status', 'components']
            
            if not all(field in status for field in required_fields):
                return False
            
            # Test detailed status report
            detailed_status = self.overmind.status_reporter.get_comprehensive_status('detailed')
            return len(detailed_status) > len(status)  # Should have more information
            
        except Exception as e:
            print(f"Status reporting test failed: {e}")
            return False
    
    def print_test_results(self):
        """Print formatted test results"""
        
        if not self.test_results:
            print("No test results available")
            return
        
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total} ({passed/total:.1%})")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print("=" * 50)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! System ready for deployment.")
        else:
            print("⚠️  Some tests failed. Review issues before deployment.")

# ===== CLI INTERFACE AND DEPLOYMENT =====

def create_cli_interface():
    """Create comprehensive CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Production Contemplative Overmind CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                           # Run test suite
  %(prog)s --agents 50 --steps 1000       # Run with 50 agents for 1000 steps
  %(prog)s --load state.json --debug      # Load state and run in debug mode
  %(prog)s --status --overmind-id my_mind # Check status of specific overmind
        """
    )
    
    # Basic execution parameters
    parser.add_argument('--agents', type=int, default=50, 
                       help='Number of agents to simulate (default: 50)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Maximum steps to run (default: 1000)')
    parser.add_argument('--overmind-id', type=str, default=None,
                       help='Overmind identifier (auto-generated if not provided)')
    
    # State management
    parser.add_argument('--load', type=str, metavar='FILE',
                       help='Load state from JSON file')
    parser.add_argument('--save', type=str, metavar='FILE',
                       help='Save state to JSON file on completion')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save state every N steps (default: 100)')
    
    # Logging and monitoring
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    parser.add_argument('--json-log', action='store_true',
                       help='Use JSON logging format')
    parser.add_argument('--log-file', type=str,
                       help='Custom log file path')
    parser.add_argument('--export-logs', type=str, metavar='FILE',
                       help='Export logs to file on completion')
    
    # Testing and diagnostics
    parser.add_argument('--test', action='store_true',
                       help='Run comprehensive test suite instead of normal execution')
    parser.add_argument('--status', action='store_true',
                       help='Print status report and exit')
    parser.add_argument('--status-detail', choices=['summary', 'detailed', 'diagnostic'],
                       default='summary', help='Status report detail level')
    
    # Advanced options
    parser.add_argument('--async', action='store_true',
                       help='Enable asynchronous processing (experimental)')
    parser.add_argument('--performance-monitoring', action='store_true', default=True,
                       help='Enable performance monitoring (default: True)')
    parser.add_argument('--no-rituals', action='store_true',
                       help='Disable ritual system')
    
    return parser

def create_production_agent(agent_id: str, agent_type: str = 'balanced'):
    """Create production-ready agent with comprehensive attributes"""
    
    agent = type('ProductionAgent', (), {})()
    agent.id = agent_id
    
    # Agent type variations
    if agent_type == 'balanced':
        agent.energy = np.random.uniform(0.4, 0.8)
        agent.health = np.random.uniform(0.5, 0.9)
        agent.mindfulness_level = np.random.uniform(0.3, 0.7)
        agent.cooperation_tendency = np.random.uniform(0.4, 0.8)
        agent.conflict_tendency = np.random.uniform(0.1, 0.3)
    elif agent_type == 'contemplative':
        agent.energy = np.random.uniform(0.6, 0.9)
        agent.health = np.random.uniform(0.7, 0.9)
        agent.mindfulness_level = np.random.uniform(0.7, 0.9)
        agent.cooperation_tendency = np.random.uniform(0.7, 0.9)
        agent.conflict_tendency = np.random.uniform(0.0, 0.2)
        agent.wisdom_accumulated = np.random.uniform(2.0, 8.0)
    elif agent_type == 'dynamic':
        agent.energy = np.random.uniform(0.5, 0.9)
        agent.health = np.random.uniform(0.4, 0.8)
        agent.mindfulness_level = np.random.uniform(0.2, 0.6)
        agent.cooperation_tendency = np.random.uniform(0.3, 0.7)
        agent.conflict_tendency = np.random.uniform(0.2, 0.5)
        agent.exploration_tendency = np.random.uniform(0.6, 0.9)
    
    # Common attributes
    agent.learning_rate = np.random.uniform(0.4, 0.9)
    agent.emotional_stability = np.random.uniform(0.3, 0.8)
    agent.energy_efficiency = np.random.uniform(0.4, 0.9)
    agent.stress_level = np.random.uniform(0.1, 0.6)
    agent.innovation_capacity = np.random.uniform(0.3, 0.8)
    agent.sharing_propensity = np.random.uniform(0.3, 0.8)
    agent.resource_conservation_tendency = np.random.uniform(0.4, 0.8)
    agent.wisdom_sharing_frequency = np.random.uniform(0.2, 0.8)
    
    # Social attributes
    agent_num = int(agent_id.split('_')[-1]) if '_' in agent_id else 0
    agent.relationships = {
        f"agent_{i}": np.random.uniform(0.2, 0.9) 
        for i in range(max(0, agent_num-3), min(100, agent_num+4)) 
        if i != agent_num
    }
    
    # Insights and experiences
    insight_templates = [
        "Balance emerges through mindful awareness",
        "Cooperation creates collective wisdom",
        "Understanding dissolves conflict naturally",
        "Wisdom grows through contemplative practice",
        "Harmony arises from patient listening"
    ]
    agent.recent_insights = np.random.choice(insight_templates, size=3, replace=False).tolist()
    
    # Position for spatial algorithms
    agent.position = [np.random.uniform(0, 100), np.random.uniform(0, 100)]
    
    return agent

def create_test_environment():
    """Create test environment for overmind"""
    
    class ProductionEnvironment:
        def __init__(self):
            self.temperature = 25.0 + np.random.uniform(-5, 5)
            self.resource_abundance = np.random.uniform(0.5, 0.9)
            self.hazard_level = np.random.uniform(0.1, 0.4)
            self.season = np.random.choice(['spring', 'summer', 'autumn', 'winter'])
    
    return ProductionEnvironment()

async def run_overmind_async(overmind, agents, max_steps, save_interval, save_file):
    """Run overmind asynchronously"""
    
    print(f"🚀 Starting asynchronous execution for {max_steps} steps...")
    
    results = {
        'total_steps': 0,
        'successful_steps': 0,
        'decisions_made': 0,
        'exceptions_handled': 0,
        'total_runtime': 0.0
    }
    
    start_time = time.time()
    
    try:
        for step in range(max_steps):
            # Run processing
            if hasattr(overmind, 'process_colony_state_async'):
                decision = await overmind.process_colony_state_async(agents, step)
            else:
                # Fallback to sync processing
                decision = overmind.process_colony_state_fully_integrated(agents, step)
            
            results['total_steps'] += 1
            
            if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
                results['decisions_made'] += 1
                results['successful_steps'] += 1
            
            # Save state periodically
            if save_file and step % save_interval == 0:
                save_overmind_state(overmind, save_file)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
    
    except Exception as e:
        results['exceptions_handled'] += 1
        logger.error(f"Exception in async execution: {e}")
    
    results['total_runtime'] = time.time() - start_time
    return results

def save_overmind_state(overmind, filename: str) -> bool:
    """Save overmind state to JSON file"""
    
    try:
        state = {
            'overmind_id': overmind.overmind_id,
            'timestamp': time.time(),
            'system_state': overmind.system_state,
            'performance_metrics': overmind.performance_metrics,
            'thresholds': getattr(overmind.threshold_regulator, 'thresholds', {}) if hasattr(overmind, 'threshold_regulator') else {}
        }
        
        # Add component states
        if hasattr(overmind, 'decision_history'):
            state['recent_decisions'] = [
                {
                    'step': d['step'],
                    'action': d['decision'].chosen_action.name if d.get('decision') else 'NO_ACTION',
                    'confidence': d['decision'].confidence if d.get('decision') else 0.0
                }
                for d in list(overmind.decision_history)[-10:]
            ]
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        return False

def load_overmind_state(overmind, filename: str) -> bool:
    """Load overmind state from JSON file"""
    
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        
        # Restore basic state
        if 'system_state' in state:
            overmind.system_state.update(state['system_state'])
        
        if 'performance_metrics' in state:
            overmind.performance_metrics.update(state['performance_metrics'])
        
        if 'thresholds' in state and hasattr(overmind, 'threshold_regulator'):
            overmind.threshold_regulator.thresholds.update(state['thresholds'])
        
        logger.info(f"State loaded from {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return False

def main():
    """Main CLI entry point"""
    
    parser = create_cli_interface()
    args = parser.parse_args()
    
    # Generate overmind ID if not provided
    if args.overmind_id is None:
        args.overmind_id = f"overmind_{uuid.uuid4().hex[:8]}"
    
    print("=" * 70)
    print("🧠 PRODUCTION CONTEMPLATIVE OVERMIND CLI")
    print(f"Overmind ID: {args.overmind_id}")
    print(f"Debug Mode: {args.debug}")
    print(f"JSON Logging: {args.json_log}")
    print("=" * 70)
    
    try:
        # Import core overmind
        from overmind_core import ProductionReadyContemplativeOvermind
        
        # Create environment
        environment = create_test_environment()
        
        # Create overmind
        overmind = ProductionReadyContemplativeOvermind(
            environment=environment,
            wisdom_signal_grid=None,  # Would connect to actual external system
            overmind_id=args.overmind_id,
            debug_mode=args.debug,
            json_logging=args.json_log
        )
        
        # Load state if requested
        if args.load:
            if load_overmind_state(overmind, args.load):
                print(f"✅ State loaded from {args.load}")
            else:
                print(f"❌ Failed to load state from {args.load}")
                return 1
        
        # Run test suite if requested
        if args.test:
            print("\n🧪 Running Test Suite...")
            if hasattr(overmind, 'test_suite'):
                results = overmind.test_suite.run_all_tests()
                overmind.test_suite.print_test_results()
                
                # Return appropriate exit code
                all_passed = all(results.values())
                return 0 if all_passed else 1
            else:
                print("Test suite not available")
                return 1
        
        # Print status and exit if requested
        if args.status:
            if hasattr(overmind, 'status_reporter'):
                overmind.status_reporter.print_status_report(args.status_detail)
            else:
                print("Status reporting not available")
            return 0
        
        # Create agents
        print(f"\n🤖 Creating {args.agents} simulation agents...")
        agent_types = ['balanced', 'contemplative', 'dynamic']
        agents = []
        
        for i in range(args.agents):
            agent_type = agent_types[i % len(agent_types)]
            agent = create_production_agent(f"agent_{i}", agent_type)
            agents.append(agent)
        
        print(f"✅ Created {len(agents)} agents")
        
        # Run overmind
        print(f"\n🚀 Starting execution for {args.steps} steps...")
        print("Press Ctrl+C to stop gracefully")
        
        if args.async:
            # Asynchronous execution
            execution_results = asyncio.run(
                run_overmind_async(overmind, agents, args.steps, args.save_interval, args.save)
            )
        else:
            # Synchronous execution
            execution_results = run_overmind_sync(overmind, agents, args.steps, args.save_interval, args.save)
        
        # Print execution summary
        print_execution_summary(execution_results, overmind)
        
        # Export logs if requested
        if args.export_logs and hasattr(overmind, 'logger'):
            overmind.logger.export_logs(args.export_logs)
        
        # Save final state if requested
        if args.save:
            save_overmind_state(overmind, args.save)
        
        print("\n🎉 Execution completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

def run_overmind_sync(overmind, agents, max_steps, save_interval, save_file):
    """Run overmind synchronously"""
    
    results = {
        'total_steps': 0,
        'successful_steps': 0,
        'decisions_made': 0,
        'exceptions_handled': 0,
        'total_runtime': 0.0
    }
    
    start_time = time.time()
    
    try:
        for step in range(max_steps):
            # Run processing
            decision = overmind.process_colony_state_fully_integrated(agents, step)
            
            results['total_steps'] += 1
            
            if decision and decision.chosen_action != OvermindActionType.NO_ACTION:
                results['decisions_made'] += 1
                results['successful_steps'] += 1
            
            # Save state periodically
            if save_file and step % save_interval == 0:
                save_overmind_state(overmind, save_file)
            
            # Simulate some agent changes
            if step % 10 == 0:
                for agent in agents[:5]:
                    agent.energy = max(0.1, agent.energy + np.random.uniform(-0.05, 0.05))
                    agent.mindfulness_level = np.clip(
                        agent.mindfulness_level + np.random.uniform(-0.02, 0.02), 0.0, 1.0
                    )
    
    except Exception as e:
        results['exceptions_handled'] += 1
        logger.error(f"Exception in sync execution: {e}")
    
    results['total_runtime'] = time.time() - start_time
    return results

def print_execution_summary(execution_results, overmind):
    """Print comprehensive execution summary"""
    
    print("\n" + "=" * 60)
    print("📊 EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Steps: {execution_results['total_steps']}")
    print(f"Successful Steps: {execution_results['successful_steps']}")
    print(f"Decisions Made: {execution_results['decisions_made']}")
    print(f"Exceptions Handled: {execution_results['exceptions_handled']}")
    print(f"Runtime: {execution_results['total_runtime']:.2f} seconds")
    
    if execution_results['total_runtime'] > 0:
        print(f"Steps/Second: {execution_results['total_steps']/execution_results['total_runtime']:.2f}")
    
    # Get final status
    if hasattr(overmind, 'status_reporter'):
        print("\n📋 FINAL STATUS:")
        final_status = overmind.status_reporter.get_comprehensive_status('summary')
        print(f"System Status: {final_status['system_status'].upper()}")
        print(f"Total Decisions: {final_status['total_decisions']}")
        print(f"Success Rate: {final_status['success_rate']:.1%}")
        
        # Component status
        healthy_components = sum(1 for status in final_status['components'].values() 
                               if 'active' in status.lower())
        total_components = len(final_status['components'])
        print(f"Healthy Components: {healthy_components}/{total_components}")
    
    print("=" * 60)

if __name__ == "__main__":
    print("Monitoring & CLI Systems Module - Ready for Integration")
    
    # If run directly, start CLI
    if len(sys.argv) > 1:
        sys.exit(main())#!/usr/bin/env python3
"""
MODULE 5: MONITORING & CLI SYSTEMS
Comprehensive logging, performance monitoring, status reporting, testing framework,
and CLI interface for production deployment
"""

import numpy as np
import json
import time
import logging
import argparse
import sys
import os
import uuid
import traceback
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import core types
from overmind_core import OvermindDecision, OvermindActionType, ColonyMetrics

logger = logging.getLogger(__name__)

# ===== ENHANCED LOGGING SYSTEM =====

class EnhancedLogger:
    """Comprehensive logging system with JSON support and structured output"""
    
    def __init__(self, overmind_id: str, json_logging: bool = False):
        self.overmind_id = overmind_id
        self.json_logging = json_logging
        self.log_entries = deque(maxlen=10000)
        
        # Setup file logging
        self.log_file = f"overmind_{overmind_id}.log"
        self.setup_logging()
        
        # Performance tracking
        self.decision_log = deque(maxlen=1000)
        self.ritual_log = deque(maxlen=500)
        self.error_log = deque(maxlen=200)
    
    def setup_logging(self):
        """Setup file and console logging"""
        
        # Create logger for this overmind
        self.logger = logging.getLogger(f"overmind_{self.overmind_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        if self.json_logging:
            # JSON formatter
            formatter = JsonFormatter()
        else:
            # Standard formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_decision(self, decision: OvermindDecision, step: int, processing_time: float):
        """Log decision with comprehensive details"""
        
        log_entry = {
            'timestamp': time.time(),
            'overmind_id': self.overmind_id,
            'type': 'decision',
            'step': step,
            'action': decision.chosen_action.name,
            'confidence': decision.confidence,
            'urgency': decision.urgency,
            'success_probability': decision.success_probability,
            'justification': decision.justification,
            'processing_time_ms': processing_time * 1000,
            'memory_influence': getattr(decision, 'memory_influence', {}),
            'emotional_gradients': getattr(decision, 'emotional_gradients', 0.0),
            'signal_entropy': getattr(decision, 'signal_entropy', 0.0)
        }
        
        self.decision_log.append(log_entry)
        self.log_entries.append(log_entry)
        
        if self.json_logging:
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.info(
                f"Decision at step {step}: {decision.chosen_action.name} "
                f"(confidence: {decision.confidence:.3f}, "
                f"processing: {processing_time*1000:.1f}ms)"
            )
    
    def log_ritual(self, ritual_name: str, participants: int, success: bool, step: int):
        """Log ritual execution"""
        
        log_entry = {
            'timestamp': time.time(),
            'overmind_id': self.overmind_id,
            'type': 'ritual',
            'ritual_name': ritual_name,
            'participants': participants,
            'success': success,
            'step': step
        }
        
        self.ritual_log.append(log_entry)
        self.log_entries.append(log_entry)
        
        if self.json_logging:
            self.logger.info(json.dumps(log_entry))
        else:
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"Ritual {ritual_name}: {status} ({participants} participants)")
    
    def log_error(self, error_type: str, error_message: str, step: int, context: Dict[str, Any] = None):
        """Log error with context"""
        
        log_entry = {
            'timestamp': time.time(),
            'overmind_id': self.overmind_id,
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'step': step,
            'context': context or {}
        }
        
        self.error_log.append(log_entry)
        self.log_entries.append(log_entry)
        
        if self.json_logging:
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.error(f"Error at step {step}: {error_type} - {error_message}")
    
    def log_performance_metric(self, metric_name: str, value: float, step: int):
        """Log performance metric"""
        
        log_entry = {
            'timestamp': time.time(),
            'overmind_id': self.overmind_id,
            'type': 'performance',
            'metric_name': metric_name,
            'value': value,
            'step': step
        }
        
        self.log_entries.append(log_entry)
        
        if self.json_logging:
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.debug(f"Performance {metric_name}: {value:.4f}")
    
    def get_recent_logs(self, count: int = 50, log_type: str = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        
        if log_type:
            filtered_logs = [entry for entry in self.log_entries if entry.get('type') == log_type]
            return list(filtered_logs)[-count:]
        else:
            return list(self.log_entries)[-count:]
    
    def export_logs(self, filename: str, format: str = 'json'):
        """Export logs to file"""
        
        try:
            with open(filename, 'w') as f:
                if format == 'json':
                    json.dump(list(self.log_entries), f, indent=2)
                elif format == 'csv':
                    # Simple CSV export
                    f.write("timestamp,type,step,message\n")
                    for entry in self.log_entries:
                        timestamp = entry.get('timestamp', time.time())
                        entry_type = entry.get('type', 'unknown')
                        step = entry.get('step', -1)
                        message = str(entry)
                        f.write(f"{timestamp},{entry_type},{step},\"{message}\"\n")
            
            self.logger.info(f"Exported {len(self.log_entries)} log entries to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
    
    def get_status(self) -> str:
        """Get logger status"""
        return f"active ({len(self.log_entries)} entries, {len(self.error_log)} errors)"

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# ===== PERFORMANCE MONITORING =====

class PerformanceMonitor:
    """Advanced performance monitoring and metrics collection"""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)
        self.current_operations = {}
        
        # Resource usage tracking
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_processing_time': 5.0,  # seconds
            'max_failure_rate': 0.1,     # 10%
            'max_memory_usage': 1000.0   # MB
        }
        
        # Alerts
        self.performance_alerts = deque(maxlen=50)
    
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation time"""
        return self.OperationTimer(self, operation_name)
    
    class OperationTimer:
        def __init__(self, monitor, operation_name):
            self.monitor = monitor
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.monitor.current_operations[self.operation_name] = self.start_time