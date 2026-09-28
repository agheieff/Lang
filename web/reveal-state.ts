export class SentenceRevealState {
  private readonly revealed: Set<string>;

  constructor(previouslyRevealed: Iterable<string> = []) {
    this.revealed = new Set(previouslyRevealed);
  }

  has(sentenceKey: string): boolean {
    return this.revealed.has(sentenceKey);
  }

  reveal(sentenceKey: string): boolean {
    if (this.revealed.has(sentenceKey)) return false;
    this.revealed.add(sentenceKey);
    return true;
  }
}

export function shouldHandleSentenceClick(
  selectionMode: boolean,
  targetIsTerm: boolean,
  targetIsControl: boolean,
): boolean {
  if (targetIsControl) return false;
  return selectionMode || !targetIsTerm;
}
