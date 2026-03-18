// chess_movegen.h — C API for fast bitboard move generation
#pragma once
#include <stdint.h>

#ifdef _WIN32
  #ifdef CHESS_MOVEGEN_EXPORTS
    #define MOVEGEN_API __declspec(dllexport)
  #else
    #define MOVEGEN_API __declspec(dllimport)
  #endif
#else
  #define MOVEGEN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Move encoding: from(6) | to(6) | promo(4) | flags(4)  packed in uint32
// promo: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
// flags: 1=capture, 2=ep, 4=castling, 8=promotion
typedef uint32_t CMove;

#define CMOVE_FROM(m)   ((int)((m) & 0x3F))
#define CMOVE_TO(m)     ((int)(((m) >> 6) & 0x3F))
#define CMOVE_PROMO(m)  ((int)(((m) >> 12) & 0xF))
#define CMOVE_FLAGS(m)  ((int)(((m) >> 16) & 0xF))

#define CMOVE_FLAG_CAPTURE   1
#define CMOVE_FLAG_EP        2
#define CMOVE_FLAG_CASTLE    4
#define CMOVE_FLAG_PROMOTION 8

static inline CMove cmove_make(int from, int to, int promo, int flags) {
    return (uint32_t)from | ((uint32_t)to << 6) | ((uint32_t)promo << 12) | ((uint32_t)flags << 16);
}

// Piece types (matching python-chess)
#define PAWN   1
#define KNIGHT 2
#define BISHOP 3
#define ROOK   4
#define QUEEN  5
#define KING   6

struct CBoard {
    uint64_t pieces[2][7];    // [color][piece_type], index 0 unused
    uint64_t occupied;
    uint64_t occupied_co[2];  // 0=white, 1=black
    int      side_to_move;    // 0=white, 1=black
    int      ep_square;       // -1 if none
    uint8_t  castling;        // bit0=WK, bit1=WQ, bit2=BK, bit3=BQ
};

/// Initialize CBoard from python-chess bitboards.
/// bb14 layout: [white_pawns, white_knights, white_bishops, white_rooks,
///               white_queens, white_kings, black_pawns, black_knights,
///               black_bishops, black_rooks, black_queens, black_kings]
/// plus occupied_white, occupied_black (14 total)
MOVEGEN_API void cboard_from_bitboards(struct CBoard* b,
                                        const uint64_t* bb14,
                                        int stm, int ep, uint8_t castle);

/// Generate all legal moves. Returns count. moves[] must have space for 256.
MOVEGEN_API int cboard_legal_moves(const struct CBoard* b, CMove* moves);

/// Fill out[64] with piece info: (color<<4)|piece_type, or 0 if empty.
MOVEGEN_API void cboard_piece_array(const struct CBoard* b, uint8_t out[64]);

/// Single piece at square: returns (color<<4)|piece_type or 0.
MOVEGEN_API int cboard_piece_at(const struct CBoard* b, int sq);

/// Returns 1 if side to move is in check.
MOVEGEN_API int cboard_is_check(const struct CBoard* b);

/// Returns 1 if move is a capture.
MOVEGEN_API int cboard_is_capture(const struct CBoard* b, CMove mv);

/// Initialize attack tables. Call once at startup.
MOVEGEN_API void movegen_init(void);

#ifdef __cplusplus
}
#endif
