<script setup lang="ts">
import { computed } from 'vue';
import type { Slot } from '../api/types';
import { pieceText, slotPlayer, variantDims } from '../domain/pieces';

const props = defineProps<{
  board: Slot[];
  variant: string;
  selected?: number | null;
  highlights?: Map<number, { type: 'move' | 'capture' }>;
  actionMasks?: number[] | null;
  markers?: number[];
  readonly?: boolean;
}>();

const emit = defineEmits<{ 'cell-click': [idx: number] }>();

const dims = computed(() => variantDims(props.variant, props.board.length));
const boardStyle = computed(() => ({
  '--board-cols': String(dims.value.cols),
  '--board-rows': String(dims.value.rows),
}));

const markerSet = computed(() => new Set(props.markers ?? []));

function cellClass(idx: number): string[] {
  const slot = props.board[idx] ?? 'Empty';
  const classes = ['chess-cell'];
  if (slot === 'Hidden') classes.push('hidden');
  else if (slot === 'Empty') classes.push('empty');
  else classes.push(slotPlayer(slot) === 'Red' ? 'red' : 'black');

  if (props.selected === idx) classes.push('selected');
  if (slot === 'Hidden' && props.actionMasks?.[idx] === 1) classes.push('legal-reveal');
  const hl = props.highlights?.get(idx);
  if (hl) classes.push(hl.type === 'capture' ? 'legal-capture' : 'legal-move');
  if (markerSet.value.has(idx)) classes.push('move-marker');
  if (props.readonly) classes.push('static');
  return classes;
}

function onClick(idx: number) {
  if (!props.readonly) emit('cell-click', idx);
}
</script>

<template>
  <div class="chess-board" :style="boardStyle">
    <div v-for="(slot, idx) in board" :key="idx" :class="cellClass(idx)" @click="onClick(idx)">
      {{ pieceText(slot) }}
    </div>
  </div>
</template>
