"""The figure number."""


class FigureNumber:
    """The figure number is incremented after each plot."""
    def __init__(self):
        self._value = 1
    
    def __call__(self):
        return self._value

    def __iadd__(self, n):
        self._value += n
        return self

def main():
    n = FigureNumber()
    assert n() == 1
    n += 1
    assert n() == 2
    n += 2
    assert n() == 4

if __name__ == '__main__':
    main()


