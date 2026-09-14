<script setup lang="ts">
import { useGame } from '../composables/useGame';
import BoardGrid from './BoardGrid.vue';

const { store, onSquareClick } = useGame();
</script>

<template>
  <div class="board-wrap" :class="{ busy: store.busy }">
    <BoardGrid
      v-if="store.state"
      :board="store.state.board"
      :variant="store.state.variant"
      :selected="store.selectedSquare"
      :highlights="store.moveHighlights"
      :action-masks="store.state.action_masks"
      @cell-click="onSquareClick"
    />
    <div v-else class="board-empty">加载中…</div>
    <div v-if="store.busy" class="busy-banner">电脑思考中…</div>
  </div>
</template>
