def levenshtein_distance(s1, s2):
    if s1 == s2:
        return 0
    rows = len(s1) + 1
    cols = len(s2) + 1

    if not s1:
        return cols - 1
    if not s2:
        return rows - 1

    cur = range(cols)
    for r in range(1, rows):
        prev, cur = cur, [r] + [0] * (cols - 1)
        for c in range(1, cols):
            deletion = prev[c] + 1
            insertion = cur[c - 1] + 1.5
            edit = prev[c - 1] + (0 if s1[r - 1] == s2[c - 1] else 2)
            cur[c] = min(edit, deletion, insertion)

    return cur[-1]


def min_distance(pred):
    dist = []
    min_dist = 1e6
    similar_words = []
    with open('word_rus.txt', 'r') as file:
        for word in file.readlines():
            word = word[:-1]
            d = levenshtein_distance(pred.lower(), word)
            if min_dist > d:
                similar_words = []
                min_dist = d
                similar_words.append(word)
            elif min_dist == d:
                similar_words.append(word)
            dist.append(d)

        return min_dist, similar_words
