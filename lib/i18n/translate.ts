import { getActiveLocale, isPseudoLocale } from './config'
import { pseudoLocalize } from './pseudo-locale'

export function localizeText(text: string): string {
  if (!text) return text
  if (isPseudoLocale()) {
    return pseudoLocalize(text)
  }
  return text
}

export function currentLocale(): string {
  return getActiveLocale()
}
