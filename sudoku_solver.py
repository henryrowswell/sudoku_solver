class SudokuSolver(object):
    def __init__(self):
        self.count = 0

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if board == None or len(board) == 0:
            return None
        else:
            self.solve(board)

    def isValid(self, board, i, j, v):
        for c in range(9):
            # check row
            if board[i][c] != "x" and board[i][c] == v:
                return False
            # check col
            if board[c][j] != "x" and board[c][j] == v:
                return False
            # check box
            if board[3*(i/3)+c/3][3*(j/3)+c%3] != "x" and board[3*(i/3)+c/3][3*(j/3)+c%3] == v:
                return False
        return True

    def solve(self, board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == "x":
                    for v in range(1,10):
                        self.count += 1
                        if self.isValid(board,i,j,str(v)):
                            board[i][j] = str(v)

                            if self.solve(board):
                                return True
                            else:
                                board[i][j] = "x"
                    return False #finished looping and didn't find a valid option
        return True #is this necessary?
