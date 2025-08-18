#!/usr/bin/env python3
"""
MDPS System Orchestrator
Central system coordination, module management, and event bus control
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import weakref

logger = logging.getLogger(__name__)

@dataclass
class ModuleInfo:
    """Information about a system module"""
    name: str
    module_type: str
    status: str = "stopped"
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    health_status: str = "unknown"
    last_heartbeat: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EventMessage:
    """Event bus message"""
    event_type: str
    source: str
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    id: str = ""

class EventBus:
    """Central event bus for system communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[EventMessage] = []
        self.max_history_size = 1000
        self.event_counters: Dict[str, int] = {}
        self.async_subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, callback: Callable, async_callback: bool = False):
        """Subscribe to an event type"""
        if async_callback:
            if event_type not in self.async_subscribers:
                self.async_subscribers[event_type] = []
            self.async_subscribers[event_type].append(callback)
        else:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            
        logger.debug(f"Subscribed to {event_type} events")
        
    def unsubscribe(self, event_type: str, callback: Callable, async_callback: bool = False):
        """Unsubscribe from an event type"""
        if async_callback:
            if event_type in self.async_subscribers and callback in self.async_subscribers[event_type]:
                self.async_subscribers[event_type].remove(callback)
        else:
            if event_type in self.subscribers and callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                
        logger.debug(f"Unsubscribed from {event_type} events")
        
    def publish(self, event: EventMessage):
        """Publish an event to all subscribers"""
        try:
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)
                
            # Update counter
            self.event_counters[event.event_type] = self.event_counters.get(event.event_type, 0) + 1
            
            # Notify synchronous subscribers
            if event.event_type in self.subscribers:
                for callback in self.subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")
                        
            # Notify asynchronous subscribers
            if event.event_type in self.async_subscribers:
                asyncio.create_task(self._notify_async_subscribers(event))
                
            logger.debug(f"Published event: {event.event_type} from {event.source}")
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            
    async def _notify_async_subscribers(self, event: EventMessage):
        """Notify asynchronous subscribers"""
        if event.event_type in self.async_subscribers:
            for callback in self.async_subscribers[event.event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in async event callback: {e}")
                    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            'total_events': len(self.event_history),
            'event_counts': self.event_counters.copy(),
            'subscriber_counts': {
                'sync': {k: len(v) for k, v in self.subscribers.items()},
                'async': {k: len(v) for k, v in self.async_subscribers.items()}
            },
            'recent_events': [e.__dict__ for e in self.event_history[-10:]] if self.event_history else []
        }

class SystemOrchestrator:
    """Main system orchestrator for MDPS"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_instances: Dict[str, Any] = {}
        self.event_bus = EventBus()
        self.running = False
        self.startup_time: Optional[datetime] = None
        self.system_health = "unknown"
        self.health_check_interval = 30.0  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.module_dependencies: Dict[str, Set[str]] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        
        # System configuration
        self.max_startup_time = 300.0  # 5 minutes
        self.max_shutdown_time = 120.0  # 2 minutes
        self.emergency_shutdown = False
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Setup default modules
        self._setup_default_modules()
        
    def _setup_default_modules(self):
        """Setup default system modules"""
        default_modules = {
            "config": ModuleInfo("config", "core", dependencies=[]),
            "database": ModuleInfo("database", "core", dependencies=["config"]),
            "logging": ModuleInfo("logging", "core", dependencies=["config"]),
            "error_handling": ModuleInfo("error_handling", "core", dependencies=["logging"]),
            "data_collection": ModuleInfo("data_collection", "data", dependencies=["config", "database", "logging"]),
            "data_processing": ModuleInfo("data_processing", "data", dependencies=["data_collection", "database", "logging"]),
            "ml_engine": ModuleInfo("ml_engine", "ml", dependencies=["data_processing", "database", "logging"]),
            "strategy_engine": ModuleInfo("strategy_engine", "trading", dependencies=["ml_engine", "database", "logging"]),
            "ui": ModuleInfo("ui", "interface", dependencies=["strategy_engine", "database", "logging"]),
            "monitoring": ModuleInfo("monitoring", "core", dependencies=["database", "logging"])
        }
        
        for name, module_info in default_modules.items():
            self.modules[name] = module_info
            self.module_dependencies[name] = set(module_info.dependencies)
            
        # Calculate startup/shutdown order
        self._calculate_module_order()
        
    def _calculate_module_order(self):
        """Calculate module startup and shutdown order based on dependencies"""
        # Topological sort for startup order
        self.startup_order = self._topological_sort(self.module_dependencies)
        
        # Reverse for shutdown order
        self.shutdown_order = list(reversed(self.startup_order))
        
        logger.info(f"Module startup order: {self.startup_order}")
        logger.info(f"Module shutdown order: {self.shutdown_order}")
        
    def _topological_sort(self, dependencies: Dict[str, Set[str]]) -> List[str]:
        """Topological sort for dependency resolution"""
        # Kahn's algorithm
        in_degree = {module: 0 for module in dependencies}
        graph = {module: set() for module in dependencies}
        
        for module, deps in dependencies.items():
            for dep in deps:
                if dep in graph:
                    graph[dep].add(module)
                    in_degree[module] += 1
                    
        queue = [module for module, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(result) != len(dependencies):
            # Circular dependency detected
            logger.warning("Circular dependency detected in modules")
            return list(dependencies.keys())
            
        return result
        
    async def start_system(self) -> bool:
        """Start the entire MDPS system"""
        try:
            logger.info("Starting MDPS system...")
            self.startup_time = datetime.now()
            self.running = True
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor())
            
            # Start modules in dependency order
            for module_name in self.startup_order:
                if not await self._start_module(module_name):
                    logger.error(f"Failed to start module: {module_name}")
                    await self.stop_system()
                    return False
                    
            logger.info("MDPS system started successfully")
            self.system_health = "healthy"
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop_system()
            return False
            
    async def stop_system(self) -> bool:
        """Stop the entire MDPS system"""
        try:
            logger.info("Stopping MDPS system...")
            self.running = False
            
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop modules in reverse dependency order
            for module_name in self.shutdown_order:
                await self._stop_module(module_name)
                
            logger.info("MDPS system stopped")
            self.system_health = "stopped"
            return True
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            return False
            
    async def _start_module(self, module_name: str) -> bool:
        """Start a specific module"""
        try:
            if module_name not in self.modules:
                logger.error(f"Module not found: {module_name}")
                return False
                
            module_info = self.modules[module_name]
            
            # Check dependencies
            for dep in module_info.dependencies:
                if dep not in self.modules or self.modules[dep].status != "running":
                    logger.error(f"Module {module_name} dependency not met: {dep}")
                    return False
                    
            # Start module
            logger.info(f"Starting module: {module_name}")
            module_info.status = "starting"
            module_info.start_time = datetime.now()
            
            # In a real implementation, this would instantiate and start the actual module
            # For now, we'll simulate module startup
            await asyncio.sleep(0.1)  # Simulate startup time
            
            module_info.status = "running"
            module_info.health_status = "healthy"
            module_info.last_heartbeat = datetime.now()
            
            logger.info(f"Module {module_name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting module {module_name}: {e}")
            if module_name in self.modules:
                self.modules[module_name].status = "failed"
            return False
            
    async def _stop_module(self, module_name: str) -> bool:
        """Stop a specific module"""
        try:
            if module_name not in self.modules:
                return True
                
            module_info = self.modules[module_name]
            
            if module_info.status == "stopped":
                return True
                
            logger.info(f"Stopping module: {module_name}")
            module_info.status = "stopping"
            
            # In a real implementation, this would gracefully stop the actual module
            # For now, we'll simulate module shutdown
            await asyncio.sleep(0.1)  # Simulate shutdown time
            
            module_info.status = "stopped"
            module_info.stop_time = datetime.now()
            module_info.health_status = "stopped"
            
            logger.info(f"Module {module_name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping module {module_name}: {e}")
            return False
            
    async def _health_monitor(self):
        """Monitor system health"""
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error
                
    async def _check_system_health(self):
        """Check overall system health"""
        try:
            healthy_modules = 0
            total_modules = len(self.modules)
            
            for module_name, module_info in self.modules.items():
                if module_info.status == "running":
                    # Check module heartbeat
                    if module_info.last_heartbeat:
                        time_since_heartbeat = (datetime.now() - module_info.last_heartbeat).total_seconds()
                        if time_since_heartbeat > self.health_check_interval * 2:
                            module_info.health_status = "unresponsive"
                            logger.warning(f"Module {module_name} is unresponsive")
                        else:
                            module_info.health_status = "healthy"
                            healthy_modules += 1
                    else:
                        module_info.health_status = "unknown"
                        
            # Update system health
            if healthy_modules == total_modules:
                self.system_health = "healthy"
            elif healthy_modules > total_modules * 0.8:
                self.system_health = "degraded"
            else:
                self.system_health = "critical"
                
            # Publish health status event
            health_event = EventMessage(
                event_type="system.health",
                source="orchestrator",
                data={
                    "system_health": self.system_health,
                    "healthy_modules": healthy_modules,
                    "total_modules": total_modules,
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.event_bus.publish(health_event)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            
    def get_module_status(self, module_name: str) -> Optional[ModuleInfo]:
        """Get status of a specific module"""
        return self.modules.get(module_name)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'running': self.running,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'system_health': self.system_health,
            'modules': {
                name: {
                    'status': info.status,
                    'health': info.health_status,
                    'start_time': info.start_time.isoformat() if info.start_time else None,
                    'last_heartbeat': info.last_heartbeat.isoformat() if info.last_heartbeat else None
                }
                for name, info in self.modules.items()
            },
            'performance_metrics': self.performance_metrics,
            'event_bus_stats': self.event_bus.get_event_stats()
        }
        
    def register_module(self, name: str, module_type: str, dependencies: List[str] = None):
        """Register a new module with the orchestrator"""
        if name in self.modules:
            logger.warning(f"Module {name} already registered")
            return False
            
        module_info = ModuleInfo(
            name=name,
            module_type=module_type,
            dependencies=dependencies or []
        )
        
        self.modules[name] = module_info
        self.module_dependencies[name] = set(dependencies or [])
        
        # Recalculate module order
        self._calculate_module_order()
        
        logger.info(f"Registered module: {name}")
        return True
        
    def unregister_module(self, name: str):
        """Unregister a module"""
        if name not in self.modules:
            return False
            
        # Remove from dependencies
        del self.modules[name]
        del self.module_dependencies[name]
        
        # Recalculate module order
        self._calculate_module_order()
        
        logger.info(f"Unregistered module: {name}")
        return True
        
    def get_event_bus(self) -> EventBus:
        """Get the system event bus"""
        return self.event_bus
        
    def publish_event(self, event_type: str, source: str, data: Dict[str, Any] = None, 
                     target: str = None, priority: int = 0):
        """Publish an event to the system event bus"""
        event = EventMessage(
            event_type=event_type,
            source=source,
            target=target,
            data=data or {},
            priority=priority
        )
        self.event_bus.publish(event)
        
    def subscribe_to_event(self, event_type: str, callback: Callable, async_callback: bool = False):
        """Subscribe to system events"""
        self.event_bus.subscribe(event_type, callback, async_callback)
        
    def unsubscribe_from_event(self, event_type: str, callback: Callable, async_callback: bool = False):
        """Unsubscribe from system events"""
        self.event_bus.unsubscribe(event_type, callback, async_callback)
        
    def emergency_shutdown(self):
        """Trigger emergency shutdown"""
        logger.critical("EMERGENCY SHUTDOWN TRIGGERED")
        self.emergency_shutdown = True
        
        # Publish emergency event
        emergency_event = EventMessage(
            event_type="system.emergency_shutdown",
            source="orchestrator",
            data={"reason": "emergency_shutdown_triggered"},
            priority=1000
        )
        self.event_bus.publish(emergency_event)
        
        # Stop system asynchronously
        asyncio.create_task(self.stop_system())
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return self.performance_metrics.copy()
        
    def update_performance_metric(self, metric_name: str, value: Any, module: str = "system"):
        """Update a performance metric"""
        if module not in self.performance_metrics:
            self.performance_metrics[module] = {}
            
        self.performance_metrics[module][metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }

# Global orchestrator instance
orchestrator = SystemOrchestrator()

def get_orchestrator() -> SystemOrchestrator:
    """Get global orchestrator instance"""
    return orchestrator

def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    return orchestrator.get_event_bus()

if __name__ == "__main__":
    # Test orchestrator functionality
    async def test_orchestrator():
        # Start system
        success = await orchestrator.start_system()
        if success:
            print("System started successfully")
            
            # Get system status
            status = orchestrator.get_system_status()
            print("System Status:", json.dumps(status, indent=2, default=str))
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Stop system
            await orchestrator.stop_system()
            print("System stopped")
        else:
            print("Failed to start system")
            
    # Run test
    asyncio.run(test_orchestrator())