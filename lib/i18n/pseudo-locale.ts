const ACCENT_MAP: Record<string, string> = {
  a: 'à',
  b: 'ƀ',
  c: 'ç',
  d: 'ð',
  e: 'é',
  f: 'ƒ',
  g: 'ğ',
  h: 'ħ',
  i: 'í',
  j: 'ĵ',
  k: 'ķ',
  l: 'ľ',
  m: 'ṁ',
  n: 'ñ',
  o: 'ô',
  p: 'þ',
  q: 'ǫ',
  r: 'ř',
  s: 'š',
  t: 'ŧ',
  u: 'ū',
  v: 'ṽ',
  w: 'ŵ',
  x: 'ẋ',
  y: 'ý',
  z: 'ž',
}

function accentuate(char: string): string {
  const lower = char.toLowerCase()
  const mapped = ACCENT_MAP[lower]
  if (!mapped) return char
  return char === lower ? mapped : mapped.toUpperCase()
}

export function pseudoLocalize(input: string): string {
  if (!input) return input

  let transformed = ''
  let inPlaceholder = false

  for (const char of input) {
    if (char === '{' || char === '}' || char === '%') {
      inPlaceholder = char === '{' ? true : char === '}' ? false : inPlaceholder
      transformed += char
      continue
    }

    transformed += inPlaceholder || /\s/.test(char) ? char : accentuate(char)
  }

  const padded = transformed.replace(/\s+/g, (match) => `${match}·`)
  return `⟦${padded}⟧`
}
