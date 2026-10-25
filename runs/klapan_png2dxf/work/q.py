import pickle, sys
prims = pickle.load(open("prims.pkl", "rb"))
x0, y0, x1, y1 = map(float, sys.argv[1:5])
ml = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
out = []
for p in prims:
    if p[0] == "line":
        a, b = p[1], p[2]
        if all(x0 <= v[0] <= x1 and y0 <= v[1] <= y1 for v in (a, b)):
            L = ((a - b) ** 2).sum() ** 0.5
            if L >= ml:
                out.append(f"L ({a[0]:.2f},{a[1]:.2f})-({b[0]:.2f},{b[1]:.2f}) len={L:.2f}")
    else:
        c, r = p[1], p[2]
        if x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and r * 6.283 * p[5] / 360 >= ml:
            out.append(f"A c=({c[0]:.2f},{c[1]:.2f}) r={r:.2f} a={p[3]:.0f}..{p[4]:.0f} sweep={p[5]:.0f}")
open("q.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(len(out))
