export function byId<T extends HTMLElement>(id: string, documentRoot: Document = document): T {
  const element = documentRoot.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
}

export function textElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text: string,
  documentRoot: Document = document,
): HTMLElementTagNameMap[K] {
  const element = documentRoot.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

export function plural(value: number, singular: string): string {
  return value === 1 ? singular : `${singular}s`;
}
