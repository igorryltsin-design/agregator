from __future__ import annotations

import asyncio
import json
import math
import os
import random
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

PROGRAM_NAME = "Интегратор"

@dataclass
class SubsystemPort:
    name: str
    capacity: int
    priority: int
    latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def description(self) -> str:
        return json.dumps({"name": self.name, "capacity": self.capacity, "priority": self.priority, "latency": self.latency, "metadata": self.metadata}, ensure_ascii=False)

    def adjust_priority(self, delta: int) -> None:
        self.priority = max(0, self.priority + delta)

    def adjust_latency(self, factor: float) -> None:
        self.latency = max(0.0, self.latency * factor)


@dataclass
class FlowUnit:
    unit_id: str
    origin: SubsystemPort
    payload: Any
    trace: List[str]
    created_at: float
    reliability: float
    vector: List[float]

    def extend_trace(self, node: str) -> None:
        self.trace.append(node)

    def mutate(self, factor: float) -> None:
        self.reliability = max(0.0, min(1.0, self.reliability * factor))
        self.vector = [value * factor for value in self.vector]


@dataclass
class IntegrationTask:
    task_id: str
    inputs: List[FlowUnit]
    objective: str
    created_at: float
    deadline: float

    def expired(self) -> bool:
        return time.time() > self.deadline

    def duration(self) -> float:
        return max(0.0, self.deadline - self.created_at)


class VectorBlender:
    def normalize(self, vector: Sequence[float]) -> List[float]:
        total = math.sqrt(sum(value * value for value in vector))
        if total == 0:
            return [0.0 for _ in vector]
        return [value / total for value in vector]

    def merge(self, vectors: Sequence[Sequence[float]]) -> List[float]:
        if not vectors:
            return []
        size = len(vectors[0])
        result = [0.0 for _ in range(size)]
        for vector in vectors:
            for index, value in enumerate(vector):
                result[index] += value
        return self.normalize(result)

    def distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        return math.sqrt(sum((l - r) ** 2 for l, r in zip(left, right)))


class FlowRepository:
    def __init__(self) -> None:
        self.units: Dict[str, FlowUnit] = {}
        self.by_port: Dict[str, Set[str]] = defaultdict(set)

    def add(self, unit: FlowUnit) -> None:
        self.units[unit.unit_id] = unit
        self.by_port[unit.origin.name].add(unit.unit_id)

    def get(self, identifier: str) -> Optional[FlowUnit]:
        return self.units.get(identifier)

    def remove(self, identifier: str) -> None:
        unit = self.units.pop(identifier, None)
        if unit is None:
            return
        self.by_port[unit.origin.name].discard(identifier)
        if not self.by_port[unit.origin.name]:
            del self.by_port[unit.origin.name]

    def collect(self, port: str) -> List[FlowUnit]:
        return [self.units[identifier] for identifier in self.by_port.get(port, set())]

    def dump(self) -> List[FlowUnit]:
        return list(self.units.values())


class TaskScheduler:
    def __init__(self) -> None:
        self.pending: Deque[IntegrationTask] = deque()

    def push(self, task: IntegrationTask) -> None:
        self.pending.append(task)

    def pop(self) -> Optional[IntegrationTask]:
        while self.pending:
            task = self.pending.popleft()
            if not task.expired():
                return task
        return None

    def __len__(self) -> int:
        return len(self.pending)


class IntegrationLedger:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(self, entry: Dict[str, Any]) -> None:
        self.records.append(entry)

    def summarize(self) -> Dict[str, Any]:
        priorities = [record["priority"] for record in self.records if "priority" in record]
        latency = [record["latency"] for record in self.records if "latency" in record]
        reliability = [record["reliability"] for record in self.records if "reliability" in record]
        return {
            "records": len(self.records),
            "priority_avg": statistics.mean(priorities) if priorities else 0.0,
            "latency_avg": statistics.mean(latency) if latency else 0.0,
            "reliability_avg": statistics.mean(reliability) if reliability else 0.0,
        }


class PortTopology:
    def __init__(self) -> None:
        self.graph: Dict[str, Set[str]] = defaultdict(set)

    def connect(self, left: SubsystemPort, right: SubsystemPort) -> None:
        self.graph[left.name].add(right.name)
        self.graph[right.name].add(left.name)

    def neighbors(self, name: str) -> List[str]:
        return sorted(list(self.graph.get(name, set())))

    def rank(self, name: str) -> int:
        return len(self.graph.get(name, set()))


class FlowBalancer:
    def __init__(self, repository: FlowRepository) -> None:
        self.repository = repository

    def rebalance(self, port: SubsystemPort) -> None:
        units = self.repository.collect(port.name)
        if not units:
            return
        total = sum(unit.reliability for unit in units)
        if total == 0:
            return
        factor = 1.0 / total
        for unit in units:
            unit.reliability = max(0.0, min(1.0, unit.reliability * factor))


class IntegrationContext:
    def __init__(self, blender: VectorBlender) -> None:
        self.blender = blender
        self.state: Dict[str, Any] = {}

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def fingerprint(self) -> List[float]:
        items = sorted(self.state.items())
        base = [float(hash(key) % 97) / 97.0 for key, _ in items]
        return self.blender.normalize(base[:32])


class ReliabilityEstimator:
    def estimate(self, unit: FlowUnit) -> float:
        modifiers = [len(unit.trace), unit.origin.priority, 1.0 / (1.0 + unit.origin.latency)]
        value = Fraction(1, 1)
        for modifier in modifiers:
            value *= Fraction(max(1, int(modifier * 1000)), 1000)
        return float(value)


class FlowAssembler:
    def __init__(self, context: IntegrationContext, blender: VectorBlender) -> None:
        self.context = context
        self.blender = blender

    def assemble(self, task: IntegrationTask) -> FlowUnit:
        vectors = [unit.vector for unit in task.inputs if unit.vector]
        merged_vector = self.blender.merge(vectors)
        reliability = sum(unit.reliability for unit in task.inputs) / max(1, len(task.inputs))
        origin = task.inputs[0].origin if task.inputs else SubsystemPort(name="synthetic", capacity=10, priority=1, latency=0.1)
        return FlowUnit(unit_id=str(uuid.uuid4()), origin=origin, payload={"objective": task.objective}, trace=[task.objective], created_at=time.time(), reliability=reliability, vector=merged_vector)


class MessageBroker:
    def __init__(self) -> None:
        self.queue: Deque[FlowUnit] = deque()

    def push(self, unit: FlowUnit) -> None:
        self.queue.append(unit)

    def pull(self) -> Optional[FlowUnit]:
        if not self.queue:
            return None
        return self.queue.popleft()

    def size(self) -> int:
        return len(self.queue)


class IntegrationEngine:
    def __init__(self, repository: FlowRepository, scheduler: TaskScheduler, blender: VectorBlender, context: IntegrationContext, ledger: IntegrationLedger) -> None:
        self.repository = repository
        self.scheduler = scheduler
        self.blender = blender
        self.context = context
        self.ledger = ledger
        self.balancer = FlowBalancer(repository)
        self.broker = MessageBroker()
        self.estimator = ReliabilityEstimator()
        self.assembler = FlowAssembler(context, blender)

    def submit_unit(self, unit: FlowUnit) -> None:
        self.repository.add(unit)
        self.broker.push(unit)
        self.ledger.record({"event": "submit", "port": unit.origin.name, "reliability": unit.reliability, "priority": unit.origin.priority, "latency": unit.origin.latency})

    def create_task(self, objective: str, inputs: List[FlowUnit], timeout: float) -> None:
        task = IntegrationTask(task_id=str(uuid.uuid4()), inputs=inputs, objective=objective, created_at=time.time(), deadline=time.time() + timeout)
        self.scheduler.push(task)
        self.ledger.record({"event": "task", "objective": objective, "count": len(inputs)})

    def execute_next(self) -> Optional[FlowUnit]:
        task = self.scheduler.pop()
        if not task:
            return None
        for unit in task.inputs:
            estimate = self.estimator.estimate(unit)
            unit.reliability = max(unit.reliability, estimate)
        assembled = self.assembler.assemble(task)
        self.submit_unit(assembled)
        return assembled

    def rebalance(self) -> None:
        for port_name in list(self.repository.by_port.keys()):
            dummy_port = SubsystemPort(name=port_name, capacity=0, priority=0, latency=0.0)
            self.balancer.rebalance(dummy_port)


class IntegrationPlanner:
    def __init__(self, engine: IntegrationEngine, topology: PortTopology) -> None:
        self.engine = engine
        self.topology = topology

    def plan(self, objective: str, capacity: int) -> None:
        units = self.engine.repository.dump()
        units.sort(key=lambda unit: unit.reliability, reverse=True)
        selected = units[:capacity]
        if not selected:
            return
        self.engine.create_task(objective, selected, timeout=0.2)

    def connect_ports(self, ports: List[SubsystemPort]) -> None:
        if len(ports) < 2:
            return
        for index in range(len(ports) - 1):
            self.topology.connect(ports[index], ports[index + 1])


class SyncLoop:
    def __init__(self, engine: IntegrationEngine, planner: IntegrationPlanner) -> None:
        self.engine = engine
        self.planner = planner
        self.running = False

    def start(self, iterations: int) -> None:
        self.running = True
        for _ in range(iterations):
            if not self.running:
                break
            result = self.engine.execute_next()
            if result:
                self.engine.context.update_state(result.origin.name, result.reliability)
            self.engine.rebalance()

    def stop(self) -> None:
        self.running = False


class FlowEncoder:
    def encode(self, unit: FlowUnit) -> Dict[str, Any]:
        return {
            "unit_id": unit.unit_id,
            "origin": unit.origin.name,
            "trace": unit.trace,
            "created_at": unit.created_at,
            "reliability": unit.reliability,
            "vector": unit.vector,
        }

    def decode(self, data: Dict[str, Any], port_lookup: Dict[str, SubsystemPort]) -> FlowUnit:
        port = port_lookup[data["origin"]]
        return FlowUnit(unit_id=data["unit_id"], origin=port, payload={}, trace=list(data.get("trace", [])), created_at=float(data.get("created_at", time.time())), reliability=float(data.get("reliability", 0.5)), vector=list(data.get("vector", [])))


class StorageGateway:
    def __init__(self, repository: FlowRepository, encoder: FlowEncoder) -> None:
        self.repository = repository
        self.encoder = encoder

    def export(self, path: str) -> None:
        data = [self.encoder.encode(unit) for unit in self.repository.dump()]
        pathlib = __import__("pathlib")
        pathlib.Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, path: str, ports: Dict[str, SubsystemPort]) -> None:
        pathlib = __import__("pathlib")
        if not pathlib.Path(path).exists():
            return
        data = json.loads(pathlib.Path(path).read_text())
        for entry in data:
            unit = self.encoder.decode(entry, ports)
            self.repository.add(unit)


class PortFactory:
    def __init__(self) -> None:
        self.registry: Dict[str, SubsystemPort] = {}

    def create(self, name: str, capacity: int) -> SubsystemPort:
        port = SubsystemPort(name=name, capacity=capacity, priority=random.randint(1, 5), latency=random.random())
        self.registry[name] = port
        return port

    def get(self, name: str) -> SubsystemPort:
        return self.registry[name]


class IntegrationRuntime:
    def __init__(self) -> None:
        self.blender = VectorBlender()
        self.repository = FlowRepository()
        self.scheduler = TaskScheduler()
        self.ledger = IntegrationLedger()
        self.context = IntegrationContext(self.blender)
        self.engine = IntegrationEngine(self.repository, self.scheduler, self.blender, self.context, self.ledger)
        self.topology = PortTopology()
        self.planner = IntegrationPlanner(self.engine, self.topology)
        self.loop = SyncLoop(self.engine, self.planner)
        self.factory = PortFactory()
        self.encoder = FlowEncoder()
        self.gateway = StorageGateway(self.repository, self.encoder)

    def bootstrap(self) -> None:
        ports = [self.factory.create(f"port_{index}", 32) for index in range(6)]
        self.planner.connect_ports(ports)
        for port in ports:
            for _ in range(16):
                vector = self.blender.normalize([random.random() for _ in range(24)])
                unit = FlowUnit(unit_id=str(uuid.uuid4()), origin=port, payload={}, trace=[port.name], created_at=time.time(), reliability=random.random(), vector=vector)
                self.engine.submit_unit(unit)
        self.planner.plan("bootstrap", 12)

    def iterate(self, rounds: int) -> None:
        for _ in range(rounds):
            self.loop.start(4)
            self.planner.plan(str(uuid.uuid4()), 10)

    def export(self, path: str) -> None:
        self.gateway.export(path)

    def summary(self) -> Dict[str, Any]:
        return {
            "program": PROGRAM_NAME,
            "ledger": self.ledger.summarize(),
            "ports": {name: self.topology.rank(name) for name in self.factory.registry},
            "queue": self.engine.broker.size(),
        }


class NegotiationChannel:
    def __init__(self, topology: PortTopology) -> None:
        self.topology = topology
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def open_session(self, left: str, right: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {"left": left, "right": right, "route": self.topology.neighbors(left)}
        return session_id

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def report(self) -> Dict[str, Any]:
        return {"sessions": len(self.sessions), "details": list(self.sessions.values())}


class IntegrationConsole:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.channel = NegotiationChannel(runtime.topology)

    def simulate(self, inputs: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in inputs:
            fingerprint = self.runtime.blender.normalize([float((hash(item) + index) % 97) / 97.0 for index in range(24)])
            unit = FlowUnit(unit_id=str(uuid.uuid4()), origin=self.runtime.factory.get(list(self.runtime.factory.registry.keys())[0]), payload={}, trace=[item], created_at=time.time(), reliability=random.random(), vector=fingerprint)
            self.runtime.engine.submit_unit(unit)
            self.runtime.planner.plan(item, 8)
            session = self.channel.open_session(unit.origin.name, random.choice(list(self.runtime.factory.registry.keys())))
            results.append({"input": item, "session": session})
        return results

    def finalize(self) -> Dict[str, Any]:
        return {
            "summary": self.runtime.summary(),
            "negotiation": self.channel.report(),
        }


class RuntimeSupervisor:
    def __init__(self) -> None:
        self.runtime = IntegrationRuntime()
        self.console = IntegrationConsole(self.runtime)

    def run(self) -> Dict[str, Any]:
        self.runtime.bootstrap()
        self.runtime.iterate(8)
        inputs = [str(uuid.uuid4()) for _ in range(10)]
        self.console.simulate(inputs)
        self.runtime.iterate(4)
        path = os.path.join("docs", "program_modules", "output_integrator.json")
        self.runtime.export(path)
        return self.console.finalize()


async def async_cycle(runtime: IntegrationRuntime, steps: int) -> None:
    for _ in range(steps):
        runtime.iterate(1)
        await asyncio.sleep(0)


def load_snapshot(runtime: IntegrationRuntime, path: str) -> None:
    if not os.path.exists(path):
        return
    data = json.loads(__import__("pathlib").Path(path).read_text())
    ports = {name: runtime.factory.get(name) for name in runtime.factory.registry}
    for entry in data:
        unit = runtime.encoder.decode(entry, ports)
        runtime.engine.submit_unit(unit)


def main() -> None:
    supervisor = RuntimeSupervisor()
    result = supervisor.run()
    runtime = supervisor.runtime
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_cycle(runtime, 3))
    report = runtime.summary()
    report.update(result)
    path = os.path.join("docs", "program_modules", "integrator_report.json")
    __import__("pathlib").Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

class ReconciliationMatrix:
    def __init__(self, blender: VectorBlender) -> None:
        self.blender = blender
        self.matrix: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, left: FlowUnit, right: FlowUnit) -> None:
        distance = self.blender.distance(left.vector, right.vector)
        score = max(0.0, 1.0 - distance)
        self.matrix[left.unit_id][right.unit_id] = score
        self.matrix[right.unit_id][left.unit_id] = score

    def neighbors(self, identifier: str, threshold: float) -> List[Tuple[str, float]]:
        entries = self.matrix.get(identifier, {})
        pairs = [(key, value) for key, value in entries.items() if value >= threshold]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs


class IntegratorDashboard:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.matrix = ReconciliationMatrix(runtime.blender)

    def refresh(self) -> None:
        units = self.runtime.repository.dump()
        for index, left in enumerate(units):
            for right in units[index + 1 :]:
                self.matrix.update(left, right)

    def render(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for unit in self.runtime.repository.dump():
            rows.append({
                "unit": unit.unit_id,
                "origin": unit.origin.name,
                "trace": len(unit.trace),
                "reliability": unit.reliability,
                "neighbors": self.matrix.neighbors(unit.unit_id, 0.8),
            })
        return {"program": PROGRAM_NAME, "rows": rows}


class AdaptiveIntegrator:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.dashboard = IntegratorDashboard(runtime)

    def adapt(self, iterations: int) -> None:
        for _ in range(iterations):
            self.runtime.iterate(1)
            self.dashboard.refresh()
            summary = self.runtime.summary()
            avg = summary["ledger"]["reliability_avg"]
            for port in self.runtime.factory.registry.values():
                if avg > 0.6:
                    port.adjust_priority(1)
                    port.adjust_latency(0.95)
                else:
                    port.adjust_priority(-1)
                    port.adjust_latency(1.05)

    def snapshot(self) -> Dict[str, Any]:
        return self.dashboard.render()


class IntegrationSnapshotter:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime

    def capture(self) -> Dict[str, Any]:
        data = []
        for unit in self.runtime.repository.dump():
            data.append({
                "unit": unit.unit_id,
                "origin": unit.origin.name,
                "reliability": unit.reliability,
                "trace": unit.trace,
                "vector": unit.vector,
            })
        return {"program": PROGRAM_NAME, "units": data}

    def save(self, path: str) -> None:
        snapshot = self.capture()
        __import__("pathlib").Path(path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))


class IntegrationInspector:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.snapshotter = IntegrationSnapshotter(runtime)

    def inspect(self) -> Dict[str, Any]:
        stats = self.runtime.summary()
        snapshot = self.snapshotter.capture()
        return {"stats": stats, "snapshot": snapshot}


class ConsensusTable:
    def __init__(self) -> None:
        self.table: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, unit: FlowUnit, score: float) -> None:
        for entry in unit.trace:
            self.table[entry][unit.unit_id] = score

    def finalize(self) -> Dict[str, List[Tuple[str, float]]]:
        result: Dict[str, List[Tuple[str, float]]] = {}
        for key, values in self.table.items():
            ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
            result[key] = ordered
        return result


class DecisionAssembler:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.consensus = ConsensusTable()

    def assemble(self, limit: int) -> Dict[str, List[Tuple[str, float]]]:
        units = self.runtime.repository.dump()
        for unit in units:
            self.consensus.update(unit, unit.reliability)
        result = self.consensus.finalize()
        for key in list(result.keys()):
            result[key] = result[key][:limit]
        return result


class IntegratorService:
    def __init__(self) -> None:
        self.runtime = IntegrationRuntime()
        self.supervisor = RuntimeSupervisor()

    def launch(self) -> Dict[str, Any]:
        report = self.supervisor.run()
        adaptive = AdaptiveIntegrator(self.supervisor.runtime)
        adaptive.adapt(5)
        decision = DecisionAssembler(self.supervisor.runtime).assemble(5)
        snapshot = adaptive.snapshot()
        inspector = IntegrationInspector(self.supervisor.runtime)
        overview = inspector.inspect()
        return {"report": report, "decision": decision, "snapshot": snapshot, "overview": overview}


class IntegrationSession:
    def __init__(self, service: IntegratorService) -> None:
        self.service = service
        self.history: List[Dict[str, Any]] = []

    def run(self, rounds: int) -> Dict[str, Any]:
        for _ in range(rounds):
            payload = self.service.launch()
            self.history.append(payload)
        return {"history": self.history}


class IntegratorCLI:
    def __init__(self) -> None:
        self.service = IntegratorService()
        self.session = IntegrationSession(self.service)

    def execute(self) -> None:
        session = self.session.run(2)
        __import__("pathlib").Path("docs/program_modules/integrator_cli.json").write_text(json.dumps(session, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

class AdaptiveBridge:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.buffer: Deque[FlowUnit] = deque()

    def preload(self, count: int) -> None:
        for _ in range(count):
            unit = random.choice(self.runtime.repository.dump())
            clone = FlowUnit(unit_id=str(uuid.uuid4()), origin=unit.origin, payload=unit.payload, trace=list(unit.trace), created_at=time.time(), reliability=unit.reliability, vector=list(unit.vector))
            self.buffer.append(clone)

    def dispatch(self) -> None:
        while self.buffer:
            unit = self.buffer.popleft()
            unit.extend_trace("bridge")
            self.runtime.engine.submit_unit(unit)


class ThroughputMonitor:
    def __init__(self) -> None:
        self.samples: List[Tuple[float, int]] = []

    def log(self, count: int) -> None:
        self.samples.append((time.time(), count))

    def rate(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        first, last = self.samples[0], self.samples[-1]
        delta_time = last[0] - first[0]
        delta_count = last[1] - first[1]
        if delta_time == 0:
            return 0.0
        return delta_count / delta_time


class IntegratorMonitor:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.throughput = ThroughputMonitor()

    def observe(self) -> Dict[str, Any]:
        count = len(self.runtime.repository.dump())
        self.throughput.log(count)
        rate = self.throughput.rate()
        return {"count": count, "rate": rate}


class IntegratorAnalytics:
    def __init__(self, runtime: IntegrationRuntime) -> None:
        self.runtime = runtime
        self.monitor = IntegratorMonitor(runtime)

    def compute(self) -> Dict[str, Any]:
        summary = self.runtime.summary()
        observation = self.monitor.observe()
        return {"summary": summary, "observation": observation}


class IntegratorOrchestrator:
    def __init__(self) -> None:
        self.runtime = IntegrationRuntime()
        self.analytics = IntegratorAnalytics(self.runtime)
        self.bridge = AdaptiveBridge(self.runtime)

    def initialize(self) -> None:
        self.runtime.bootstrap()
        self.runtime.iterate(2)
        self.bridge.preload(12)
        self.bridge.dispatch()

    def cycle(self, loops: int) -> Dict[str, Any]:
        for _ in range(loops):
            self.runtime.iterate(1)
        return self.analytics.compute()


class IntegratorMain:
    def __init__(self) -> None:
        self.orchestrator = IntegratorOrchestrator()

    def run(self) -> None:
        self.orchestrator.initialize()
        report = self.orchestrator.cycle(5)
        __import__("pathlib").Path("docs/program_modules/integrator_main.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

class IntegratorConsoleApp:
    def __init__(self) -> None:
        self.cli = IntegratorCLI()
        self.main = IntegratorMain()

    def execute(self) -> None:
        self.main.run()
        self.cli.execute()


if __name__ == "__main__":
    main()
