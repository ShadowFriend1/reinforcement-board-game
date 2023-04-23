#! /usr/bin/env python
"""
 Project: Python Chess
 File name: ChessGUI_pygame.py
 Description:  Uses pygame (http://www.pygame.org/) to draw the
	chess board, as well as get user input through mouse clicks.
	The chess tile graphics were taken from Wikimedia Commons, 
	http://commons.wikimedia.org/wiki/File:Chess_tile_pd.png

 Copyright (C) 2009 Steve Osborne, srosborne (at) gmail.com
 http://yakinikuman.wordpress.com/
 """

import os
import sys

import pygame
from pygame.locals import *


def ConvertToAlgebraicNotation_row(row):
    # (row,col) format used in Python Chess code starts at (0,0) in the upper left.
    # Algebraic notation starts in the lower left and uses "a..h" for the column.
    B = ['8', '7', '6', '5', '4', '3', '2', '1']
    return B[row]


def ConvertToAlgebraicNotation_col(col):
    # (row,col) format used in Python Chess code starts at (0,0) in the upper left.
    # Algebraic notation starts in the lower left and uses "a..h" for the column.
    A = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    return A[col]


class ChessGUI_pygame:
    def __init__(self, graphicStyle=1, image_dir='images'):
        os.environ['SDL_VIDEO_CENTERED'] = '1'  # should center pygame window on the screen
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((500, 500))
        self.boardStart_x = 50
        self.boardStart_y = 50
        pygame.display.set_caption('Python Chess')
        self.image_dir = image_dir
        self.LoadImages(graphicStyle)
        # pygame.font.init() - should be already called by pygame.init()
        self.fontDefault = pygame.font.Font(None, 20)

    def LoadImages(self, graphicStyle):
        if graphicStyle == 0:
            self.square_size = 50  # all images must be images 50 x 50 pixels
            self.white_square = pygame.image.load(os.path.join(self.image_dir, "white_square.png")).convert()
            self.brown_square = pygame.image.load(os.path.join(self.image_dir, "brown_square.png")).convert()
            self.cyan_square = pygame.image.load(os.path.join(self.image_dir, "cyan_square.png")).convert()
            # "convert()" is supposed to help pygame display the images faster.  It seems to mess up transparency - makes it all black!
            # And, for this chess program, the images don't need to change that fast.
            self.black_pawn = pygame.image.load(os.path.join(self.image_dir, "blackPawn.png"))
            self.black_rook = pygame.image.load(os.path.join(self.image_dir, "blackRook.png"))
            self.black_knight = pygame.image.load(os.path.join(self.image_dir, "blackKnight.png"))
            self.black_bishop = pygame.image.load(os.path.join(self.image_dir, "blackBishop.png"))
            self.black_king = pygame.image.load(os.path.join(self.image_dir, "blackKing.png"))
            self.black_queen = pygame.image.load(os.path.join(self.image_dir, "blackQueen.png"))
            self.white_pawn = pygame.image.load(os.path.join(self.image_dir, "whitePawn.png"))
            self.white_rook = pygame.image.load(os.path.join(self.image_dir, "whiteRook.png"))
            self.white_knight = pygame.image.load(os.path.join(self.image_dir, "whiteKnight.png"))
            self.white_bishop = pygame.image.load(os.path.join(self.image_dir, "whiteBishop.png"))
            self.white_king = pygame.image.load(os.path.join(self.image_dir, "whiteKing.png"))
            self.white_queen = pygame.image.load(os.path.join(self.image_dir, "whiteQueen.png"))
        elif graphicStyle == 1:
            self.square_size = 50
            self.white_square = pygame.image.load(os.path.join(self.image_dir, "white_square.png")).convert()
            self.brown_square = pygame.image.load(os.path.join(self.image_dir, "brown_square.png")).convert()
            self.cyan_square = pygame.image.load(os.path.join(self.image_dir, "cyan_square.png")).convert()

            self.black_pawn = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_pd.png")).convert()
            self.black_pawn = pygame.transform.scale(self.black_pawn, (self.square_size, self.square_size))
            self.black_rook = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_rd.png")).convert()
            self.black_rook = pygame.transform.scale(self.black_rook, (self.square_size, self.square_size))
            self.black_knight = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_nd.png")).convert()
            self.black_knight = pygame.transform.scale(self.black_knight, (self.square_size, self.square_size))
            self.black_bishop = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_bd.png")).convert()
            self.black_bishop = pygame.transform.scale(self.black_bishop, (self.square_size, self.square_size))
            self.black_king = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_kd.png")).convert()
            self.black_king = pygame.transform.scale(self.black_king, (self.square_size, self.square_size))
            self.black_queen = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_qd.png")).convert()
            self.black_queen = pygame.transform.scale(self.black_queen, (self.square_size, self.square_size))

            self.white_pawn = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_pl.png")).convert()
            self.white_pawn = pygame.transform.scale(self.white_pawn, (self.square_size, self.square_size))
            self.white_rook = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_rl.png")).convert()
            self.white_rook = pygame.transform.scale(self.white_rook, (self.square_size, self.square_size))
            self.white_knight = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_nl.png")).convert()
            self.white_knight = pygame.transform.scale(self.white_knight, (self.square_size, self.square_size))
            self.white_bishop = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_bl.png")).convert()
            self.white_bishop = pygame.transform.scale(self.white_bishop, (self.square_size, self.square_size))
            self.white_king = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_kl.png")).convert()
            self.white_king = pygame.transform.scale(self.white_king, (self.square_size, self.square_size))
            self.white_queen = pygame.image.load(os.path.join(self.image_dir, "Chess_tile_ql.png")).convert()
            self.white_queen = pygame.transform.scale(self.white_queen, (self.square_size, self.square_size))

    def ConvertToScreenCoords(self, chessSquareTuple):
        # converts a (row,col) chessSquare into the pixel location of the upper-left corner of the square
        (row, col) = chessSquareTuple
        screenX = self.boardStart_x + col * self.square_size
        screenY = self.boardStart_y + row * self.square_size
        return (screenX, screenY)

    def ConvertToChessCoords(self, screenPositionTuple):
        # converts a screen pixel location (X,Y) into a chessSquare tuple (row,col)
        # x is horizontal, y is vertical
        # (x=0,y=0) is upper-left corner of the screen
        (X, Y) = screenPositionTuple
        row = (Y - self.boardStart_y) / self.square_size
        col = (X - self.boardStart_x) / self.square_size
        return (row, col)

    def Draw(self, board, highlightSquares=[]):
        self.screen.fill((0, 0, 0))
        boardSize = len(
            board)  # board should be square.  boardSize should be always 8 for chess, but I dislike "magic numbers" :)

        # draw blank board
        current_square = 0
        for r in range(boardSize):
            for c in range(boardSize):
                (screenX, screenY) = self.ConvertToScreenCoords((r, c))
                if current_square:
                    self.screen.blit(self.brown_square, (screenX, screenY))
                    current_square = (current_square + 1) % 2
                else:
                    self.screen.blit(self.white_square, (screenX, screenY))
                    current_square = (current_square + 1) % 2

            current_square = (current_square + 1) % 2

        # draw row/column labels around the edge of the board
        # chessboard_obj = ChessBoard(0)  # need a dummy object to access some of ChessBoard's methods....
        color = (255, 255, 255)  # white
        antialias = 1

        # top and bottom - display cols
        for c in range(boardSize):
            for r in [-1, boardSize]:
                (screenX, screenY) = self.ConvertToScreenCoords((r, c))
                screenX = screenX + self.square_size / 2
                screenY = screenY + self.square_size / 2
                notation = ConvertToAlgebraicNotation_col(c)
                renderedLine = self.fontDefault.render(notation, antialias, color)
                self.screen.blit(renderedLine, (screenX, screenY))

        # left and right - display rows
        for r in range(boardSize):
            for c in [-1, boardSize]:
                (screenX, screenY) = self.ConvertToScreenCoords((r, c))
                screenX = screenX + self.square_size / 2
                screenY = screenY + self.square_size / 2
                notation = ConvertToAlgebraicNotation_row(r)
                renderedLine = self.fontDefault.render(notation, antialias, color)
                self.screen.blit(renderedLine, (screenX, screenY))

        # highlight squares if specified
        for square in highlightSquares:
            (screenX, screenY) = self.ConvertToScreenCoords(square)
            self.screen.blit(self.cyan_square, (screenX, screenY))

        # draw pieces
        for r in range(boardSize):
            for c in range(boardSize):
                (screenX, screenY) = self.ConvertToScreenCoords((r, c))
                if board[r][c] == 'bP':
                    self.screen.blit(self.black_pawn, (screenX, screenY))
                if board[r][c] == 'bR':
                    self.screen.blit(self.black_rook, (screenX, screenY))
                if board[r][c] == 'bT':
                    self.screen.blit(self.black_knight, (screenX, screenY))
                if board[r][c] == 'bB':
                    self.screen.blit(self.black_bishop, (screenX, screenY))
                if board[r][c] == 'bQ':
                    self.screen.blit(self.black_queen, (screenX, screenY))
                if board[r][c] == 'bK':
                    self.screen.blit(self.black_king, (screenX, screenY))
                if board[r][c] == 'wP':
                    self.screen.blit(self.white_pawn, (screenX, screenY))
                if board[r][c] == 'wR':
                    self.screen.blit(self.white_rook, (screenX, screenY))
                if board[r][c] == 'wT':
                    self.screen.blit(self.white_knight, (screenX, screenY))
                if board[r][c] == 'wB':
                    self.screen.blit(self.white_bishop, (screenX, screenY))
                if board[r][c] == 'wQ':
                    self.screen.blit(self.white_queen, (screenX, screenY))
                if board[r][c] == 'wK':
                    self.screen.blit(self.white_king, (screenX, screenY))

        pygame.display.flip()

    def EndGame(self, board):
        self.Draw(board)  # draw board to show end game status
        pygame.event.set_blocked(MOUSEMOTION)
        while 1:
            e = pygame.event.wait()
            if e.type is KEYDOWN:
                pygame.quit()
                sys.exit(0)
            if e.type is QUIT:
                pygame.quit()
                sys.exit(0)

    def GetPlayerInput(self, board, legal_moves):
        # returns ((from_row,from_col),(to_row,to_col))
        fromSquareChosen = 0
        toSquareChosen = 0
        while not fromSquareChosen or not toSquareChosen:
            squareClicked = []
            pygame.event.set_blocked(MOUSEMOTION)
            e = pygame.event.wait()
            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    fromSquareChosen = 0
                    fromTuple = []
            if e.type == MOUSEBUTTONDOWN:
                (mouseX, mouseY) = pygame.mouse.get_pos()
                squareClicked = self.ConvertToChessCoords((mouseX, mouseY))
                squareClicked = tuple([int(element) for element in squareClicked])
                if squareClicked[0] < 0 or squareClicked[0] > 7 or squareClicked[1] < 0 or squareClicked[1] > 7:
                    squareClicked = []  # not a valid chess square
            if e.type == QUIT:  # the "x" kill button
                pygame.quit()

            if not fromSquareChosen and not toSquareChosen:
                self.Draw(board)
                if squareClicked != []:
                    (r, c) = squareClicked
                    r = int(r)
                    c = int(c)
                    legal = False
                    possibleDestinations = []
                    for i in legal_moves:
                        if i[0] == squareClicked:
                            legal = True
                            possibleDestinations.append(i[1])
                    if legal:
                        fromSquareChosen = 1
                        fromTuple = squareClicked

            elif fromSquareChosen and not toSquareChosen:
                # implements castling
                if fromTuple in [(7, 4), (0, 4)]:
                    if ((8, 0), (0, 0)) in legal_moves:
                        possibleDestinations.append((fromTuple[0], fromTuple[1] - 2))
                    elif ((8, 0), (0, 1)) in legal_moves:
                        possibleDestinations.append((fromTuple[0], fromTuple[1] + 2))
                self.Draw(board, possibleDestinations)
                if squareClicked != []:
                    if squareClicked in possibleDestinations:
                        toSquareChosen = 1
                        toTuple = squareClicked
                        # Returns relevant from and to value to indicate castling to environment
                        if fromTuple in [(7, 4), (0, 4)]:
                            if toTuple == (fromTuple[0], fromTuple[1] - 2):
                                fromTuple = (8, 0)
                                toTuple = (0, 0)
                            elif toTuple == (fromTuple[0], fromTuple[1] + 2):
                                fromTuple = (8, 0)
                                toTuple = (0, 1)
                    else:  # blank square or opposite color piece not in possible destinations clicked
                        fromSquareChosen = 0
                        possibleDestinations = []
        return (fromTuple, toTuple)

    def GetClickedSquare(self, mouseX, mouseY):
        # test function
        print("User clicked screen position x =", mouseX, "y =", mouseY)
        (row, col) = self.ConvertToChessCoords((mouseX, mouseY))
        if 8 > col >= 0 and 8 > row >= 0:
            print("  Chess board units row =", row, "col =", col)

    def TestRoutine(self):
        # test function
        pygame.event.set_blocked(MOUSEMOTION)
        while 1:
            e = pygame.event.wait()
            if e.type is QUIT:
                return
            if e.type is KEYDOWN:
                if e.key is K_ESCAPE:
                    pygame.quit()
                    return
            if e.type is MOUSEBUTTONDOWN:
                (mouseX, mouseY) = pygame.mouse.get_pos()
                # x is horizontal, y is vertical
                # (x=0,y=0) is upper-left corner of the screen
                self.GetClickedSquare(mouseX, mouseY)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # try out some development / testing stuff if this file is run directly
    testBoard = [['bR', 'bT', 'bB', 'bQ', 'bK', 'bB', 'bT', 'bR'],
                 ['bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP'],
                 ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                 ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                 ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                 ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                 ['wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP'],
                 ['wR', 'wT', 'wB', 'wQ', 'wK', 'wB', 'wT', 'wR']]

    validSquares = [(5, 2), (1, 1), (1, 5), (7, 6)]

    game = ChessGUI_pygame()
    game.Draw(testBoard)
    game.TestRoutine()
