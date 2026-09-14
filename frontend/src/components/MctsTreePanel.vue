<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { MctsEdge } from '../api/types';
import { MCTS_MAX_CHILDREN, useMctsTree } from '../composables/useMctsTree';
import { useGame } from '../composables/useGame';
import BoardGrid from './BoardGrid.vue';

const emit = defineEmits<{ close: [] }>();

const {
  store,
  treeTransform,
  refresh,
  toggle,
  search,
  select,
  clearSelection,
  pvNodeIds,
  buildLayout,
  collectNodes,
  hasChildren,
} = useMctsTree();
const { store: game } = useGame();

// mcts-viz 风格：矩形节点 + 正交折线连线
const NW = 116;
const NH = 64;
const DX = 138;
const DY = 128;
const PAD = 70;

const wrapEl = ref<HTMLElement | null>(null);

const layout = computed(() => buildLayout());
const nodes = computed(() => (layout.value ? collectNodes(layout.value) : []));
const pvSet = computed(() => (layout.value ? pvNodeIds(layout.value) : new Set<number>()));

const svgViewBox = computed(() => {
  const leafCount = Math.max(1, nodes.value.filter((n) => n.children.length === 0).length);
  const maxDepth = Math.max(0, ...nodes.value.map((n) => n.depth));
  return { width: leafCount * DX + PAD * 2, height: (maxDepth + 1) * DY + PAD * 2 };
});

const rootInfo = computed(() => store.rootInfo);

const px = (n: { x: number }) => PAD + n.x * DX;
const py = (n: { y: number }) => PAD + n.y * DY;

function nodeTitle(n: { id: number; edge: { is_chance: boolean; chance_prob: number; action: number } | null }): string {
  if (rootInfo.value && n.id === rootInfo.value.root.id) return 'ROOT';
  if (!n.edge) return '';
  return n.edge.is_chance ? `翻 ${(n.edge.chance_prob * 100).toFixed(0)}%` : `着法 #${n.edge.action}`;
}

function edgePath(n: { x: number; y: number }, c: { x: number; y: number }): string {
  const y1 = py(n) + NH / 2;
  const y2 = py(c) - NH / 2;
  const mid = (y1 + y2) / 2;
  return `M ${px(n)} ${y1} L ${px(n)} ${mid} L ${px(c)} ${mid} L ${px(c)} ${y2}`;
}

function edgeMidY(n: { y: number }, c: { y: number }): number {
  return (py(n) + NH / 2 + (py(c) - NH / 2)) / 2;
}

function isChosenEdge(n: { edge: { is_chance: boolean; action: number } | null }): boolean {
  const info = store.rootInfo;
  return !!info && !!n.edge && !n.edge.is_chance && n.edge.action === info.chosen_action;
}

function isPvEdge(n: { id: number }, c: { id: number }): boolean {
  return pvSet.value.has(n.id) && pvSet.value.has(c.id);
}

function edgeWidth(n: { edge: { prior: number; is_chance: boolean } | null }): number {
  if (!n.edge || n.edge.is_chance) return 1.5;
  return Math.max(1.5, Math.min(7, (n.edge?.prior ?? 0) * 40));
}

function pathStroke(n: { id: number }, c: { id: number; edge: MctsEdge | null }): string {
  if (isChosenEdge(c)) return '#d32f2f';
  if (isPvEdge(n, c)) return '#0d6efd';
  return c.edge?.is_chance ? '#c9a24b' : '#9a8f7f';
}

function pathWidth(n: { id: number }, c: { id: number; edge: MctsEdge | null }): number {
  if (isChosenEdge(c)) return 4.5;
  if (isPvEdge(n, c)) return Math.max(3, edgeWidth(c) + 1.5);
  return edgeWidth(c);
}

const valueClass = (v: number) => (v > 0 ? 'val-pos' : v < 0 ? 'val-neg' : undefined);

function nodeClass(n: { id: number; edge: MctsEdge | null }): Record<string, boolean> {
  return {
    'is-chosen': isChosenEdge(n),
    'is-pv': pvSet.value.has(n.id),
    'is-selected': store.selectedId === n.id,
  };
}

async function onNodeClick(nodeId: number) {
  void select(nodeId);
  await toggle(nodeId);
}

const moveLabel = computed(() => {
  const coords = store.selectedBoard?.move_coords ?? [];
  if (coords.length === 0) return '初始局面（无前着）';
  if (coords.length === 1) return `翻开 #${coords[0]}`;
  return `#${coords[0]} → #${coords[1]}`;
});

function onWheel(evt: WheelEvent) {
  evt.preventDefault();
  const factor = evt.deltaY < 0 ? 1.15 : 1 / 1.15;
  const rect = wrapEl.value?.getBoundingClientRect();
  if (!rect) return;
  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;
  treeTransform.tx = cx - (cx - treeTransform.tx) * factor;
  treeTransform.ty = cy - (cy - treeTransform.ty) * factor;
  treeTransform.scale = Math.max(0.2, Math.min(4, treeTransform.scale * factor));
}

let dragging = false;
let lastX = 0;
let lastY = 0;

function onDragStart(evt: MouseEvent) {
  if ((evt.target as HTMLElement).closest('g')) return;
  dragging = true;
  lastX = evt.clientX;
  lastY = evt.clientY;
}
function onDragMove(evt: MouseEvent) {
  if (!dragging) return;
  treeTransform.tx += evt.clientX - lastX;
  treeTransform.ty += evt.clientY - lastY;
  lastX = evt.clientX;
  lastY = evt.clientY;
}
function onDragEnd() {
  dragging = false;
}

// 打开面板（visible 变 true）或棋盘更新且面板开启时刷新树
watch(
  () => game.state,
  () => {
    if (store.rootInfo) void refresh();
  },
);

function transformAttr(): string {
  return `translate(${treeTransform.tx}, ${treeTransform.ty}) scale(${treeTransform.scale})`;
}
</script>

<template>
  <aside class="sidebar mcts-sidebar">
    <header class="sidebar-header">
      <h3>MCTS 搜索树</h3>
      <button class="icon-button" aria-label="关闭" @click="emit('close')">×</button>
    </header>
    <div class="mcts-toolbar">
      <button :disabled="store.searching" @click="search">
        {{ store.searching ? '搜索中…' : '重新搜索' }}
      </button>
      <button @click="refresh">刷新</button>
      <label><input v-model="store.showAll" type="checkbox" /> 显示全部子节点</label>
      <span v-if="rootInfo" class="mcts-info">根 N={{ rootInfo.root.n }}
        <span :class="valueClass(rootInfo.root.q)">Q={{ rootInfo.root.q.toFixed(3) }}</span>
        选择动作 #{{ rootInfo.chosen_action }}</span>
    </div>
    <div class="mcts-body">
      <div
        ref="wrapEl"
        class="mcts-tree-zone"
        @wheel="onWheel"
        @mousedown="onDragStart"
        @mousemove="onDragMove"
        @mouseup="onDragEnd"
        @mouseleave="onDragEnd"
      >
        <div v-if="!rootInfo" class="mcts-empty">
          暂无搜索树：请先让 MctsDL / MctsOnnx 对手走一步，或手动搜索
        </div>
        <svg v-else :viewBox="`0 0 ${svgViewBox.width} ${svgViewBox.height}`" width="100%" height="100%">
          <g :transform="transformAttr()">
            <!-- 正交折线连线（mcts-viz 风格；蓝=主变例，红=根实际选择） -->
            <template v-for="n in nodes" :key="`edges-${n.id}`">
              <template v-for="c in n.children" :key="`e-${n.id}-${c.id}`">
                <path
                  :d="edgePath(n, c)"
                  fill="none"
                  :stroke="pathStroke(n, c)"
                  :stroke-width="pathWidth(n, c)"
                  :stroke-dasharray="c.edge?.is_chance ? '5 4' : undefined"
                />
                <text
                  v-if="(c.edge?.prior ?? 0) > 0"
                  :x="px(c) + 5"
                  :y="edgeMidY(n, c) - 4"
                  font-size="10"
                  fill="#8a8378"
                >
                  P={{ c.edge!.prior.toFixed(2) }} N={{ c.n }}
                </text>
              </template>
            </template>

            <!-- 矩形节点 -->
            <g
              v-for="n in nodes"
              :key="`n-${n.id}`"
              :transform="`translate(${px(n)}, ${py(n)})`"
              class="mcts-node"
              @click.stop="onNodeClick(n.id)"
            >
              <rect
                class="mcts-node-rect"
                :x="-NW / 2"
                :y="-NH / 2"
                :width="NW"
                :height="NH"
                rx="8"
                :class="nodeClass(n)"
              />
              <text :y="-6" text-anchor="middle" class="mcts-node-title">
                {{ nodeTitle(n) }}
              </text>
              <text :y="14" text-anchor="middle" class="mcts-node-stats">
                <tspan>N={{ n.n }}</tspan>
                <tspan dx="4" :class="valueClass(n.q)">Q={{ n.q.toFixed(2) }}</tspan>
              </text>
              <g v-if="hasChildren(n.id)" class="mcts-collapse">
                <circle :cy="NH / 2" r="9" />
                <text :y="NH / 2 + 4" text-anchor="middle">
                  {{ store.expanded.has(n.id) ? '−' : '+' }}
                </text>
              </g>
            </g>

            <template v-for="n in nodes" :key="`hidden-${n.id}`">
              <text
                v-if="store.expanded.has(n.id) && n.hiddenCount > 0"
                :x="px(n)"
                :y="py(n) + DY - 24"
                text-anchor="middle"
                font-size="10"
                fill="#b0651f"
              >
                …还有 {{ n.hiddenCount }} 个子节点（前 {{ MCTS_MAX_CHILDREN }} 个）
              </text>
            </template>
          </g>
        </svg>
      </div>

      <aside class="mcts-inspector">
        <h4>节点局面</h4>
        <p v-if="!store.selectedId" class="mcts-inspector-empty">
          点击左侧任意节点，在此查看该节点的棋盘与搜索统计
        </p>
        <p v-else-if="store.inspectorLoading && !store.selectedBoard" class="mcts-inspector-empty">
          加载中…
        </p>
        <template v-else-if="store.selectedBoard">
          <div class="mcts-board-host">
            <BoardGrid
              :board="store.selectedBoard.state.board"
              :variant="store.selectedBoard.state.variant"
              :markers="store.selectedBoard.move_coords"
              readonly
            />
          </div>
          <p class="mcts-move-label">到达着法：{{ moveLabel }}</p>
          <dl class="mcts-stats">
            <div>
              <dt>节点</dt>
              <dd>
                #{{ store.selectedId }}
                <span v-if="store.selectedDetail">
                  （{{ store.selectedDetail.player === 'Red' ? '红方' : '黑方' }}行棋）
                </span>
              </dd>
            </div>
            <div v-if="store.selectedDetail?.is_chance"><dt>类型</dt><dd>机会节点</dd></div>
            <div v-if="store.selectedDetail?.is_terminal"><dt>类型</dt><dd>终局</dd></div>
            <div>
              <dt>轮到</dt>
              <dd>{{ store.selectedBoard.state.current_player === 'Red' ? '红方' : '黑方' }}</dd>
            </div>
            <div>
              <dt>血量</dt>
              <dd>
                红 {{ store.selectedBoard.state.hp_red }} / 黑 {{ store.selectedBoard.state.hp_black }}
              </dd>
            </div>
            <div>
              <dt>步数</dt>
              <dd>
                {{ store.selectedBoard.state.total_step_counter }}（无吃子
                {{ store.selectedBoard.state.move_counter }}）
              </dd>
            </div>
            <template v-if="store.selectedDetail">
              <div><dt>N</dt><dd>{{ store.selectedDetail.n }}</dd></div>
              <div>
                <dt>Q</dt>
                <dd :class="valueClass(store.selectedDetail.q)">
                  {{ store.selectedDetail.q.toFixed(4) }}
                </dd>
              </div>
              <div>
                <dt>Q_hp</dt>
                <dd :class="valueClass(store.selectedDetail.health_q)">
                  {{ store.selectedDetail.health_q.toFixed(4) }}
                </dd>
              </div>
              <div><dt>prior</dt><dd>{{ store.selectedDetail.prior.toFixed(4) }}</dd></div>
              <div><dt>logit</dt><dd>{{ store.selectedDetail.logit.toFixed(3) }}</dd></div>
              <div>
                <dt>V 先验</dt>
                <dd :class="valueClass(store.selectedDetail.initial_value)">
                  {{ store.selectedDetail.initial_value.toFixed(4) }}
                </dd>
              </div>
              <div>
                <dt>子树</dt>
                <dd>
                  {{ store.selectedDetail.child_count }} 子节点 /
                  {{ store.selectedDetail.outcome_count }} 机会结果
                </dd>
              </div>
            </template>
          </dl>
          <button class="mini mcts-clear" @click="clearSelection">取消选中</button>
        </template>
      </aside>
    </div>
  </aside>
</template>
