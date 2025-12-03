import math


def rotate(x, y, deg):
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)
    return round(cos * x - sin * y, 1), round(sin * x + cos * y, 1)


print(rotate(1, 0, 90))  # → (0.0, 1.0)
