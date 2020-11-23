from math import inf
import chess
from time import time
import sys
import chess.polyglot as bookloader
from tqdm import tqdm
from threading import Thread
from time import sleep
import random
from math import exp
from multiprocessing import Process
import math
#import numba as nb

#from numba import prange 
################################### piece squre table ###################################
# piece square tables (for analysis) (change these to change the engine behavior)
#import numpy as np
bPawnTable = (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0)
wPawnTable = (0,0,0,0,0,0,0,0,
              10,8,9,5,3,6,2,5,
              11, 6,5,6,8,3,10,
              -6,5,3,60,60,4,6,1,
              10, 11, 12, 11, 10, 9, 5,3,
              12, 13, 14, 15,14, 13, 12, 12,
              70, 70, 80, 70, 70, 80 , 80, 80,
              90,90,90,90,90,90,90,90)
              
bKnightTable = ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69)
wKnightTable = bKnightTable[::-1]
bBishopTable = ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10)
wBishopTable = bBishopTable[::-1]
bRookTable = (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32)
wRookTable = bRookTable[::-1]
bQueenTable =  (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42)
wQueenTable = bQueenTable[::-1]
bKingTable = (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18)
wKingTable = bKingTable[::-1]

king_safty = (
  0,  0,   1,   2,   3,   5,   7,   9,  12,  15,
  18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
  68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
 140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
 260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
 377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
 494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 200, 500, 300, 100, 100, 100, 100, 200, 500, 400
)

tables = {"doubled":-1, "isolated": -3,"-1": bPawnTable, "1":wPawnTable, "-3.20":bKnightTable, "3.20":wKnightTable, "-3.4":bBishopTable,"3.4":wBishopTable,"5":wRookTable,"-5":bRookTable,"9":wQueenTable, "-9":bQueenTable,"-20000":bKingTable, "20000":wKingTable}
weights = {"mobility": 0.2, "piece_table":0.1, "king_table":0.01} # np.random.random_sample()

################################### /* piece squre table /* ###################################

def opening_move(board, book_path):
   try:
      return bookloader.open_reader(book_path).find(board).move
   except IndexError:
      return None
################################### evaluation ###################################

def get_piece_val(piece:str):
   temp = 0
   if piece == "k":
      return -200
   if piece == "q":
      return -9
   if piece == "r":
      return -5
   if piece == "b":
      return -3.4
   if piece == "n":
      return -3.2
   if piece == "p":
      return -1
   if piece == "K":
      return 200
   if piece == "Q":
      return 9
   if piece == "R":
      return 5
   if piece == "B":
      return 3.3
   if piece == "N":
      return 3.20
   if piece == "P":
      return 1
   return temp

def get_piece_val2(piece:str):
   policy = {"\n":0, " ":0, ".":0, "k":-200, "q":-9, "r":-5, "b":-3.4, "n":-3.2, "p":-1, "K":200, "Q":9, "R":5, "B":3.4, "N": 3.2, "P":1}
   return policy[piece]

def passed_pawn(pm, is_end_game):
    whiteYmax = [ -1 ] * 8
    blackYmin = [ 8 ] * 8

    for key, p in pm.items():
        if p.piece_type != chess.PAWN:
            continue

        x = key & 7
        y = key >> 3

        if p.color == chess.WHITE:
            whiteYmax[x] = max(whiteYmax[x], y)
        else:
            blackYmin[x] = min(blackYmin[x], y)

    scores = [ [ 0, 5, 20, 30, 40, 50, 80, 0 ], [ 0, 5, 20, 40, 70, 120, 200, 0 ] ]

    score = 0

    for key, p in pm.items():
        if p.piece_type != chess.PAWN:
            continue

        x = key & 7
        y = key >> 3

        if p.color == chess.WHITE:
            left = (x > 0 and (blackYmin[x - 1] <= y or blackYmin[x - 1] == 8)) or x == 0
            front = blackYmin[x] < y or blackYmin[x] == 8
            right = (x < 7 and (blackYmin[x + 1] <= y or blackYmin[x + 1] == 8)) or x == 7

            if left and front and right:
                score += scores[is_end_game][y]

        else:
            left = (x > 0 and (whiteYmax[x - 1] >= y or whiteYmax[x - 1] == -1)) or x == 0
            front = whiteYmax[x] > y or whiteYmax[x] == -1
            right = (x < 7 and (whiteYmax[x + 1] >= y or whiteYmax[x + 1] == -1)) or x == 7

            if left and front and right:
                score -= scores[is_end_game][7 - y] 
    return score

def evaluate(board_fen, debugger=False):
   policy = {"\n":0, " ":0, ".":0, "k":-20000, "q":-900, "r":-500, "b":-340, "n":-320, "p":-100, "K":20000, "Q":900, "R":500, "B":340, "N": 320, "P":100}
   thisboard = chess.Board(board_fen)
   if thisboard.is_variant_win(): return 10000
   if thisboard.is_variant_loss(): return -10000
   if thisboard.is_variant_draw(): return 0
   if list(thisboard.legal_moves) == []:
      if thisboard.is_check(): return 10000
      return 0
   evaluation = 0
   index = 0

   for piece in thisboard.attackers(chess.WHITE,chess.E4):
      if piece == 1:
         evaluation += 0.1
   for piece in thisboard.attackers(chess.BLACK, chess.E4):
      if piece == 1:
         evaluation -= 0.1
   
   for piece in thisboard.attackers(chess.WHITE,chess.D4):
      if piece == 1:
         evaluation += 0.1
   for piece in thisboard.attackers(chess.BLACK, chess.D4):
      if piece == 1:
         evaluation -= 0.1
   
   for piece in thisboard.attackers(chess.WHITE,chess.E5):
      if piece == 1:
         evaluation += 0.1
   for piece in thisboard.attackers(chess.BLACK, chess.E5):
      if piece == 1:
         evaluation -= 0.1

   for piece in thisboard.attackers(chess.WHITE,chess.D4):
      if piece == 1:
         evaluation += 0.1
   for piece in thisboard.attackers(chess.BLACK, chess.D4):
      if piece == 1:
         evaluation -= 0.1

   for i in str(thisboard):
      evaluation += policy[i]
      try:
         piece_val = policy[i]
         """if piece_val != 200:
            evaluation += tables[str(piece_val)][index] * weights["piece_table"]
         else:
            evaluation += tables[str(piece_val)][index] * weights["king_table"]"""
         evaluation += piece_val
      except KeyError:
         index -= 1
      index += 1
   for i in chess.LegalMoveGenerator(thisboard): evaluation += 0.1 * weights["mobility"]
   #evaluation += passed_pawn(thisboard.piece_map(), False)
   move = list(thisboard.legal_moves)[0]
   move = chess.Move.from_uci(str(move))
   thisboard.push(move)
   for i in chess.LegalMoveGenerator(thisboard): evaluation -= 0.1 * weights["mobility"]
   return evaluation

def update_nodes(pbar, st):
   while tomove:
      prev_nodesss = nodesss
      sleep(1)
      pbar.set_description("calculating %s at %s nodes/s in depth %s" % (str(move), nodesss-prev_nodesss, d))

def curr():
   p_move = None
   sleep(1)
   while True:
      if time() - st > 2 and d > 1 and p_move != move:
         p_move = move
         print("info depth {} currmove {}".format(d+1, move))

def iteretive_deepening(board, depth, alpha, beta, color, book_path = r"/Volumes/POOPOO USB/lichess-bot/engines/engine2/book/book.bin"):
   global tomove
   global move
   global nodesss
   global d
   global st
   # make dict with all move naes (uci) and then the all the move sequnces will be saved. then when we have the best one. we get the item from the dict with the name of the best uci move.
   possible = tuple(board.legal_moves) #tuple(order(board))
   st = time()
   """with tqdm(total=len(possible), unit="move") as pbar:
      nodesssss = Thread(target=update_nodes, args=[pbar, st])
      nodesssss.daemon = True
      nodesssss.start()"""
   """      for i in range(6, depth):
   depths.append(i)"""
   """t = Thread(target=curr)
   t.daemon = True
   t.start()"""
   for d in range(depth):
      value = -inf
      st = time()
      nodesss = 0
      for move in possible:
         nodesss += 1
         #assert board.is_legal(move)
         board.push(move)
         if color == 1:
            firstguess = negamax(board, d, inf, -10000, 1)
         else:
            firstguess = negamax(board, d, -inf, 10000, -1)
         board.pop()
         
         
         #print(" " + str(move), firstguess)
         """if str(move) == "e7e5":
            print(" " + str(move), firstguess)"""
         if firstguess > value:
            best_move = move
            value = firstguess
         if firstguess >= beta:
            break
         #pbar.update(1)
      if firstguess >= beta:
         break
      print("info depth {} time {} nodes {} score cp {} nps {} pv {}".format(d+1, int((time()-st)*1000), nodesss, value, nodesss/(time() - st), best_move))
      #pbar.reset()
   return best_move

nodesss = 0
fhc = 0
def order(board):
   moves = []
   captures = []
   these_caps = board.generate_pseudo_legal_captures()
   if len(tuple(these_caps)) != 0:
      for capture in these_caps:
         captures.append([str(capture), board.piece_type_at(capture.to_square) * 10 - board.piece_type_at(capture.from_square)])
      captures = reversed(sorted(captures))
      for i in captures:
         moves.append(chess.Move.from_uci(i[0]))
   for i in board.legal_moves:
      if not i in board.generate_pseudo_legal_captures():
         moves.append(i)
   return moves
def negamax(board, depth, alpha, beta, color):
   global nodesss
   """
   color can be either 1 or -1. 1 for white and -1 for black
   """
   possible = tuple(board.generate_legal_moves())
   if depth == 0 or len(possible) == 0:
      nodesss += 1
      return evaluate(board.fen()) * color #evaluate(board.fen()) * -color #quiesce(-10000, 10000, board, 1) * -color
   value = -inf
   for move in possible:
      nodesss += 1
      board.push(move)
      value = max(value, -negamax(board, depth-1, -alpha, -beta, -color))
      board.pop() # can do class node that stores a board and the possible moves and then I can say if it is a terminal node (a node that determains loss, win and draws) with a parameter purpose.
      if value > alpha:
         alpha = value
      if alpha >= beta:
         break
   return value



ffh = 0
fh = 0
#@nb.jit
def pvs_max(board, depth, alpha, beta):
   global nodesss
   possible = tuple(board.legal_moves)
   first = True
   if depth == 0 or possible == (): 
      #nodesss += 1
      return -evaluate(board.fen())
   b = beta
   a = alpha
   for move in possible:
      #nodesss += 1
      board.push(move)
      if first:
         score = pvs_min(board, depth - 1, -a, -b)
      else:
            score = pvs_min(board, depth - 1, -a, -b) # search with a null window *)
            if alpha < score < beta:
               score = pvs_min(board, depth - 1, -b, -score) # if it failed high, do a full re-search *)
      board.pop()
      alpha = max(alpha, score)
      if alpha >= beta:
         break # beta cut-off
      first = False
   return alpha

#@nb.jit
def pvs_min(board, depth, alpha, beta):
   global nodesss
   possible = tuple(board.legal_moves)
   first = True
   if depth == 0 or possible == (): 
      #nodesss += 1
      return -evaluate(board.fen())
   b = beta
   a = alpha
   for move in possible:
      #nodesss += 1

      board.push(move)

      if first:
         score = pvs_max(board, depth - 1, -a, -b)
      else:
            score = pvs_max(board, depth - 1, -a, -b) # search with a null window *)
            if alpha < score < beta:
               score = pvs_max(board, depth - 1, -a, -b) # if it failed high, do a full re-search *)

      board.pop()
      alpha = min(alpha, score)
      if alpha <= beta:
         """if first:
            global ffh
            ffh += 1
         global fh
         fh += 1"""
         break # beta cut-off
      first = False
   return alpha

class color_error(BaseException):
   def __init__(self, msg):
      raise BaseException(msg)

def quiesce(alpha,beta, board, depth):
   global nodesss
   captures = tuple(board.generate_pseudo_legal_captures())
   stand_pat = evaluate(board.fen())
   if( stand_pat >= beta ): return beta
   if( alpha < stand_pat ): alpha = stand_pat
   if depth == 0: return stand_pat
   for capture in captures:
      nodesss += 1
      board.push(capture)
      score = -quiesce( -beta, -alpha, board, depth - 1 )
      board.pop()
      if( score >= beta ): return beta
      if( score > alpha ): alpha = score
   return alpha

#def ucb1()

def mcts(board_fen, color, num_of_simulations):
   global nodesss
   board = chess.Board(board_fen)
   w = 0
   b = 0
   d = 0
   flen = len
   
   for iteration in range(num_of_simulations):
      while not board.is_game_over():
         nodesss += 1
         possible = tuple(board.legal_moves)
         board.push(possible[random.randint(0, flen(possible) - 1)])
      result = board.result()
      if result[0] == "1" and result[1] != "/":
         w += 1
      if result[0] == "0":
         b += 1
      if result[1] == "/":
         d += 1
      board.reset()
   if color == 1:
      return (w+d/2)/num_of_simulations
   if color == -1:
      return (b+d/2)/num_of_simulations

def sigmoid(x):
   return 1/(1+exp(-x))
def update_mcts(pbar):
   while tomove:
      prev_nodesss = nodesss
      sleep(1)
      pbar.set_description("calculating %s at %s nodes/s with %s simulations" % (str(move), nodesss-prev_nodesss, d))
def mcts_root(board, color, num_of_simulations):
   global tomove
   global move
   tomove = True
   global d
   this_opening = None #opening_move(board, r"E:\lichess-bot\engines\engine2\book\book.bin")
   if this_opening != None:
      return this_opening, 1
   d = num_of_simulations
   st = time()
   best_value = -inf
   possible = tuple(board.legal_moves)
   with tqdm(total=len(possible), unit="move") as pbar:
      nodesssss = Thread(target=update_mcts, args=[pbar])
      nodesssss.daemon = True
      nodesssss.start()
      for move in possible:
         board.push(move)
         value = mcts(board.fen(), color, num_of_simulations)
         #print(str(move) == "f7a2")
         if str(move) == "a6a2":
            print("",str(move), value)
         if value > best_value:  
            best_move = move
            best_value = value
         board.pop()
         pbar.update(1)
      tomove = False
      return best_move, best_value

def main2():
   opt = sys.argv
   board = chess.Board(opt[1])
   depth = int(opt[2])
   color = int(opt[3])
   iteretive_deepening(board, depth, -inf, 10000, color)
   #print(mcts_root(board, color, depth))
   #print(negamax(board, depth, -inf, 10000, 1))

def game():
   global nodesss
   results = []
   game = []
   games = []
   for _ in range(1):
      board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
      color = -1
      while not board.is_game_over():
         nodesss = 0
         move = iteretive_deepening(board, 3, -inf, 400, color)
         game.append(move)
         board.push(move)
         color = -color
         print(board)
         print(board.fen())
      results.append((board.fen(), board.result()))
      try:
        games.append(board.variation_san([m for m in game]))
      except ValueError:
         continue
      game = []
   print()
   print(results)
   open("game.pgn", "w+").write(str(games))
if __name__=="__main__":
   while True:
      i = input().split(", ")
      iteretive_deepening(chess.Board(), int(i[0]), -inf, 10000, int(i[1]))
