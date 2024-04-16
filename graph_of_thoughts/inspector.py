from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field
from typing_extensions import Self

from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
from graph_of_thoughts.operations.operations import Operation, OperationSummary


@dataclass
class OperationNode:
    id: int
    successors: list[Self]
    summary: OperationSummary


class SOTAStatus(Enum):
    RUNNING = 0
    FINISHED = 1
    FAILED = 2


@dataclass
class Cytoscape:
    elements: list[dict]
    fcose_horizontal_layout: list[list[str]]
    fcose_relative_constraints: list[dict]


class GraphSummary(BaseModel):
    nodes: dict[int, OperationSummary]
    adj_list: dict[int, list[int]]
    roots: list[int]
    status: SOTAStatus
    timestamp: datetime = Field(default=datetime.now())

    @classmethod
    def from_graph(cls, graph: GraphOfOperations, status=SOTAStatus.RUNNING) -> Self:
        assert graph.roots is not None, "The operations graph has no root"

        queue = [*graph.roots]

        nodes: dict[int, OperationSummary] = {}
        adj_set: dict[int, set[int]] = defaultdict(set)
        root_ids = [r.id for r in graph.roots]

        for operation in queue:
            nodes[operation.id] = operation.get_summary()

            for succ in operation.successors:
                adj_set[operation.id].add(succ.id)
                queue.append(succ)

        adj_list = {k: list(v) for k, v in adj_set.items()}

        return cls(nodes=nodes, adj_list=adj_list, roots=root_ids, status=status)

    def as_object_graph(self) -> list[OperationNode]:
        roots: list[OperationNode] = []

        for root_id in self.roots:
            op = OperationNode(id=root_id, successors=[], summary=self.nodes[root_id])
            roots.append(op)

        def add_successors(node: OperationNode):
            # Recursive fun
            # Inefficient, duplicates successors :(
            succ_ids = self.adj_list[node.id]

            if not succ_ids:
                return

            succesors: list[OperationNode] = []

            for id_ in succ_ids:
                node = OperationNode(id=id_, successors=[], summary=self.nodes[id_])
                succesors.append(node)

            node.successors = succesors

            for succ in succesors:
                add_successors(succ)

        return roots

    def _fcose_horizontal_layout(
        self,
    ) -> list[list[str]]:
        level_ids: list[list[str]] = []  # list 1: level, list2: level ids
        already_traversed: set[int] = set()

        def traverse_level(level_items: Iterable[int]):
            # add a new level
            level_ids.append([str(id_) for id_ in level_items])
            already_traversed.update(level_items)

            next_level: set[int] = set()

            for item in level_items:
                if item in self.adj_list:
                    next_level.update(self.adj_list[item])
            next_level_f = list(
                filter(lambda x: x not in already_traversed, next_level)
            )

            if len(next_level_f):
                traverse_level(next_level_f)

        traverse_level(self.roots)

        return level_ids

    def _fcose_rel_constraints(self, horizontal_layout: list[list[str]]) -> list[dict]:
        constraints: list[dict] = []

        for i in range(1, len(horizontal_layout)):
            prev_lay = horizontal_layout[i - 1]
            current_lay = horizontal_layout[i]

            constraints.append({"top": prev_lay[0], "bottom": current_lay[0]})

        return constraints

    def _fcose_vertical_layout(
        self, horizontal_layout: list[list[str]]
    ) -> list[list[str]]:
        layout: list[list[str]] = []

        for i in range(1, len(horizontal_layout)):
            prev_lay = horizontal_layout[i - 1]
            current_lay = horizontal_layout[i]

            layout.append(
                [prev_lay[len(prev_lay) // 2], current_lay[len(prev_lay) // 2]]
            )
        return layout

    def as_cytoscape(
        self,
        sort_by_idx: bool = True,
        selectable_edges: bool = False,
        selectable_nodes: bool = True,
    ) -> Cytoscape:
        elements = []

        # Create the element list
        for node_id_, op in self.nodes.items():
            node_id = str(node_id_)
            el = {
                "data": {"id": node_id, "background_color": op.status.get_css_color()},
                "selectable": selectable_nodes,
            }
            elements.append(el)

            neigh_ids = self.adj_list.get(node_id_)

            if not neigh_ids:
                continue

            for neigh_id in neigh_ids:
                el = {
                    "data": {
                        "source": node_id,
                        "target": neigh_id,
                        "id": f"{node_id} -> {neigh_id}",
                        "selectable": selectable_edges,
                    }
                }
                elements.append(el)

        if sort_by_idx:
            sorted(elements, key=lambda x: str(x["data"]["id"]))

        # create the
        layout = self._fcose_horizontal_layout()
        cons = self._fcose_rel_constraints(layout)
        return Cytoscape(
            elements=elements,
            fcose_horizontal_layout=layout,
            fcose_relative_constraints=cons,
        )
