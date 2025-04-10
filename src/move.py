class Move:
    def __init__(self, initial, final):
        self.initial = initial
        self.final = final

    def __str__(self):
        s = ''
        s += f'({self.initial.col}, {self.initial.row})'
        s += f' -> ({self.final.col}, {self.final.row})'
        return s

    def __eq__(self, other):
        return self.initial == other.initial and self.final == other.final

    def algebraic(self):
        """
        Convert the move to algebraic notation.
        Assumes that:
         - Columns range from 0 to 7 are mapped to files 'a' through 'h'.
         - Rows range from 0 to 7 are mapped such that row 0 is rank 8, and row 7 is rank 1.
        For example, a move from (col=4, row=6) to (col=4, row=4) will return "e2e4".
        """
        file_from = chr(ord('a') + self.initial.col)
        # Assuming row 0 is the top row (rank 8) and row 7 is the bottom (rank 1)
        rank_from = str(8 - self.initial.row)
        file_to = chr(ord('a') + self.final.col)
        rank_to = str(8 - self.final.row)
        return file_from + rank_from + file_to + rank_to
