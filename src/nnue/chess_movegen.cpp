// chess_movegen.cpp — Bitboard-based legal move generation
// Follows the same DLL pattern as nnue_inference.cpp

#include "chess_movegen.h"
#include <cstring>

// ─── Bit manipulation helpers ───
static inline int popcount(uint64_t x) {
#if defined(_MSC_VER)
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

static inline int lsb(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    return __builtin_ctzll(x);
#endif
}

static inline uint64_t lsb_bb(uint64_t x) { return x & (-x); }

// Iterator: pop least significant bit, return its index
static inline int pop_lsb(uint64_t& x) {
    int sq = lsb(x);
    x &= x - 1;
    return sq;
}

// ─── Constants ───
#define RANK_1 0x00000000000000FFULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_3 0x0000000000FF0000ULL
#define RANK_4 0x00000000FF000000ULL
#define RANK_5 0x000000FF00000000ULL
#define RANK_6 0x0000FF0000000000ULL
#define RANK_7 0x00FF000000000000ULL
#define RANK_8 0xFF00000000000000ULL
#define FILE_A 0x0101010101010101ULL
#define FILE_H 0x8080808080808080ULL

static inline int rank_of(int sq) { return sq >> 3; }
static inline int file_of(int sq) { return sq & 7; }
static inline uint64_t sq_bb(int sq) { return 1ULL << sq; }

// ─── Precomputed attack tables ───
static uint64_t KNIGHT_ATTACKS[64];
static uint64_t KING_ATTACKS[64];
static uint64_t PAWN_ATTACKS[2][64]; // [color][square]
static bool tables_initialized = false;

static void init_knight_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        uint64_t bb = sq_bb(sq);
        uint64_t attacks = 0;
        attacks |= (bb << 17) & ~FILE_A;  // up 2, right 1
        attacks |= (bb << 15) & ~FILE_H;  // up 2, left 1
        attacks |= (bb << 10) & ~(FILE_A | FILE_A << 1); // up 1, right 2
        attacks |= (bb <<  6) & ~(FILE_H | FILE_H >> 1); // up 1, left 2
        attacks |= (bb >> 17) & ~FILE_H;
        attacks |= (bb >> 15) & ~FILE_A;
        attacks |= (bb >> 10) & ~(FILE_H | FILE_H >> 1);
        attacks |= (bb >>  6) & ~(FILE_A | FILE_A << 1);
        KNIGHT_ATTACKS[sq] = attacks;
    }
}

static void init_king_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        uint64_t bb = sq_bb(sq);
        uint64_t attacks = 0;
        attacks |= (bb << 8);                     // up
        attacks |= (bb >> 8);                     // down
        attacks |= (bb << 1) & ~FILE_A;          // right
        attacks |= (bb >> 1) & ~FILE_H;          // left
        attacks |= (bb << 9) & ~FILE_A;          // up-right
        attacks |= (bb << 7) & ~FILE_H;          // up-left
        attacks |= (bb >> 7) & ~FILE_A;          // down-right
        attacks |= (bb >> 9) & ~FILE_H;          // down-left
        KING_ATTACKS[sq] = attacks;
    }
}

static void init_pawn_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        uint64_t bb = sq_bb(sq);
        // White pawn attacks (upward)
        PAWN_ATTACKS[0][sq] = ((bb << 7) & ~FILE_H) | ((bb << 9) & ~FILE_A);
        // Black pawn attacks (downward)
        PAWN_ATTACKS[1][sq] = ((bb >> 7) & ~FILE_A) | ((bb >> 9) & ~FILE_H);
    }
}

MOVEGEN_API void movegen_init(void) {
    if (tables_initialized) return;
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
    tables_initialized = true;
}

// ─── Classical sliding attacks (ray-based) ───
// Directions: N=+8, S=-8, E=+1, W=-1, NE=+9, NW=+7, SE=-7, SW=-9
static uint64_t slide_attacks(int sq, uint64_t occupied, const int* deltas, int num_deltas) {
    uint64_t attacks = 0;
    for (int d = 0; d < num_deltas; d++) {
        int s = sq;
        while (true) {
            int prev_rank = rank_of(s);
            int prev_file = file_of(s);
            s += deltas[d];
            if (s < 0 || s > 63) break;
            int new_rank = rank_of(s);
            int new_file = file_of(s);
            // Check wrapping: rank/file distance must be <= 1 per step
            int dr = new_rank - prev_rank;
            int df = new_file - prev_file;
            if (dr < -1 || dr > 1 || df < -1 || df > 1) break;
            attacks |= sq_bb(s);
            if (occupied & sq_bb(s)) break; // blocked
        }
    }
    return attacks;
}

static const int BISHOP_DELTAS[] = { 9, 7, -9, -7 };
static const int ROOK_DELTAS[]   = { 8, -8, 1, -1 };

static inline uint64_t bishop_attacks(int sq, uint64_t occupied) {
    return slide_attacks(sq, occupied, BISHOP_DELTAS, 4);
}

static inline uint64_t rook_attacks(int sq, uint64_t occupied) {
    return slide_attacks(sq, occupied, ROOK_DELTAS, 4);
}

static inline uint64_t queen_attacks(int sq, uint64_t occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

// ─── Attack detection ───
// Returns all squares attacked by the given side
static uint64_t attacks_by(const struct CBoard* b, int side) {
    uint64_t atk = 0;
    uint64_t occ = b->occupied;
    uint64_t bb;

    // Pawns
    bb = b->pieces[side][PAWN];
    while (bb) {
        int sq = pop_lsb(bb);
        atk |= PAWN_ATTACKS[side][sq];
    }

    // Knights
    bb = b->pieces[side][KNIGHT];
    while (bb) {
        int sq = pop_lsb(bb);
        atk |= KNIGHT_ATTACKS[sq];
    }

    // Bishops
    bb = b->pieces[side][BISHOP];
    while (bb) {
        int sq = pop_lsb(bb);
        atk |= bishop_attacks(sq, occ);
    }

    // Rooks
    bb = b->pieces[side][ROOK];
    while (bb) {
        int sq = pop_lsb(bb);
        atk |= rook_attacks(sq, occ);
    }

    // Queens
    bb = b->pieces[side][QUEEN];
    while (bb) {
        int sq = pop_lsb(bb);
        atk |= queen_attacks(sq, occ);
    }

    // King
    bb = b->pieces[side][KING];
    if (bb) {
        int sq = lsb(bb);
        atk |= KING_ATTACKS[sq];
    }

    return atk;
}

// Is the given square attacked by the given side?
static bool is_attacked_by(const struct CBoard* b, int sq, int side) {
    uint64_t occ = b->occupied;

    // Check pawn attacks (reverse direction)
    if (PAWN_ATTACKS[1 - side][sq] & b->pieces[side][PAWN]) return true;

    // Knight
    if (KNIGHT_ATTACKS[sq] & b->pieces[side][KNIGHT]) return true;

    // Bishop/Queen (diagonal)
    uint64_t diag = bishop_attacks(sq, occ);
    if (diag & (b->pieces[side][BISHOP] | b->pieces[side][QUEEN])) return true;

    // Rook/Queen (straight)
    uint64_t straight = rook_attacks(sq, occ);
    if (straight & (b->pieces[side][ROOK] | b->pieces[side][QUEEN])) return true;

    // King
    if (KING_ATTACKS[sq] & b->pieces[side][KING]) return true;

    return false;
}

// ─── Board initialization ───
MOVEGEN_API void cboard_from_bitboards(struct CBoard* b,
                                        const uint64_t* bb14,
                                        int stm, int ep, uint8_t castle) {
    movegen_init();
    memset(b, 0, sizeof(struct CBoard));

    // bb14 layout: [wp, wn, wb, wr, wq, wk, bp, bn, bb_, br, bq, bk, occ_w, occ_b]
    b->pieces[0][PAWN]   = bb14[0];
    b->pieces[0][KNIGHT] = bb14[1];
    b->pieces[0][BISHOP] = bb14[2];
    b->pieces[0][ROOK]   = bb14[3];
    b->pieces[0][QUEEN]  = bb14[4];
    b->pieces[0][KING]   = bb14[5];
    b->pieces[1][PAWN]   = bb14[6];
    b->pieces[1][KNIGHT] = bb14[7];
    b->pieces[1][BISHOP] = bb14[8];
    b->pieces[1][ROOK]   = bb14[9];
    b->pieces[1][QUEEN]  = bb14[10];
    b->pieces[1][KING]   = bb14[11];

    b->occupied_co[0] = bb14[12];
    b->occupied_co[1] = bb14[13];
    b->occupied = bb14[12] | bb14[13];

    b->side_to_move = stm;
    b->ep_square = ep;
    b->castling = castle;
}

// ─── Piece queries ───
MOVEGEN_API void cboard_piece_array(const struct CBoard* b, uint8_t out[64]) {
    memset(out, 0, 64);
    for (int color = 0; color < 2; color++) {
        for (int pt = PAWN; pt <= KING; pt++) {
            uint64_t bb = b->pieces[color][pt];
            while (bb) {
                int sq = pop_lsb(bb);
                out[sq] = (uint8_t)((color << 4) | pt);
            }
        }
    }
}

MOVEGEN_API int cboard_piece_at(const struct CBoard* b, int sq) {
    uint64_t mask = sq_bb(sq);
    for (int color = 0; color < 2; color++) {
        if (!(b->occupied_co[color] & mask)) continue;
        for (int pt = PAWN; pt <= KING; pt++) {
            if (b->pieces[color][pt] & mask)
                return (color << 4) | pt;
        }
    }
    return 0;
}

MOVEGEN_API int cboard_is_check(const struct CBoard* b) {
    int us = b->side_to_move;
    uint64_t king_bb = b->pieces[us][KING];
    if (!king_bb) return 0;
    int king_sq = lsb(king_bb);
    return is_attacked_by(b, king_sq, 1 - us) ? 1 : 0;
}

MOVEGEN_API int cboard_is_capture(const struct CBoard* b, CMove mv) {
    return (CMOVE_FLAGS(mv) & CMOVE_FLAG_CAPTURE) ? 1 : 0;
}

// ─── Pseudo-legal move generation ───
// Adds moves to the buffer, returns count added
static int gen_pawn_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    int them = 1 - us;
    uint64_t pawns = b->pieces[us][PAWN];
    uint64_t empty = ~b->occupied;
    uint64_t enemies = b->occupied_co[them];

    if (us == 0) { // White
        // Single push
        uint64_t push1 = (pawns << 8) & empty;
        // Double push (only from rank 2)
        uint64_t push2 = ((push1 & RANK_3) << 8) & empty;

        uint64_t promo_push = push1 & RANK_8;
        push1 &= ~RANK_8;

        while (push1) {
            int to = pop_lsb(push1);
            moves[count++] = cmove_make(to - 8, to, 0, 0);
        }
        while (push2) {
            int to = pop_lsb(push2);
            moves[count++] = cmove_make(to - 16, to, 0, 0);
        }
        while (promo_push) {
            int to = pop_lsb(promo_push);
            int from = to - 8;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION); // queen
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION); // knight
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION); // rook
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION); // bishop
        }

        // Captures
        uint64_t cap_left  = (pawns << 7) & ~FILE_H & enemies;
        uint64_t cap_right = (pawns << 9) & ~FILE_A & enemies;
        uint64_t promo_cap_left  = cap_left  & RANK_8;
        uint64_t promo_cap_right = cap_right & RANK_8;
        cap_left  &= ~RANK_8;
        cap_right &= ~RANK_8;

        while (cap_left) {
            int to = pop_lsb(cap_left);
            moves[count++] = cmove_make(to - 7, to, 0, CMOVE_FLAG_CAPTURE);
        }
        while (cap_right) {
            int to = pop_lsb(cap_right);
            moves[count++] = cmove_make(to - 9, to, 0, CMOVE_FLAG_CAPTURE);
        }
        while (promo_cap_left) {
            int to = pop_lsb(promo_cap_left);
            int from = to - 7;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
        }
        while (promo_cap_right) {
            int to = pop_lsb(promo_cap_right);
            int from = to - 9;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
        }

        // En passant
        if (b->ep_square >= 0) {
            uint64_t ep_bb = sq_bb(b->ep_square);
            uint64_t ep_left  = (pawns << 7) & ~FILE_H & ep_bb;
            uint64_t ep_right = (pawns << 9) & ~FILE_A & ep_bb;
            if (ep_left)
                moves[count++] = cmove_make(b->ep_square - 7, b->ep_square, 0,
                                             CMOVE_FLAG_CAPTURE | CMOVE_FLAG_EP);
            if (ep_right)
                moves[count++] = cmove_make(b->ep_square - 9, b->ep_square, 0,
                                             CMOVE_FLAG_CAPTURE | CMOVE_FLAG_EP);
        }
    } else { // Black
        uint64_t push1 = (pawns >> 8) & empty;
        uint64_t push2 = ((push1 & RANK_6) >> 8) & empty;

        uint64_t promo_push = push1 & RANK_1;
        push1 &= ~RANK_1;

        while (push1) {
            int to = pop_lsb(push1);
            moves[count++] = cmove_make(to + 8, to, 0, 0);
        }
        while (push2) {
            int to = pop_lsb(push2);
            moves[count++] = cmove_make(to + 16, to, 0, 0);
        }
        while (promo_push) {
            int to = pop_lsb(promo_push);
            int from = to + 8;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION);
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION);
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION);
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION);
        }

        uint64_t cap_left  = (pawns >> 7) & ~FILE_A & enemies;
        uint64_t cap_right = (pawns >> 9) & ~FILE_H & enemies;
        uint64_t promo_cap_left  = cap_left  & RANK_1;
        uint64_t promo_cap_right = cap_right & RANK_1;
        cap_left  &= ~RANK_1;
        cap_right &= ~RANK_1;

        while (cap_left) {
            int to = pop_lsb(cap_left);
            moves[count++] = cmove_make(to + 7, to, 0, CMOVE_FLAG_CAPTURE);
        }
        while (cap_right) {
            int to = pop_lsb(cap_right);
            moves[count++] = cmove_make(to + 9, to, 0, CMOVE_FLAG_CAPTURE);
        }
        while (promo_cap_left) {
            int to = pop_lsb(promo_cap_left);
            int from = to + 7;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
        }
        while (promo_cap_right) {
            int to = pop_lsb(promo_cap_right);
            int from = to + 9;
            moves[count++] = cmove_make(from, to, 4, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 1, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 3, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
            moves[count++] = cmove_make(from, to, 2, CMOVE_FLAG_PROMOTION | CMOVE_FLAG_CAPTURE);
        }

        if (b->ep_square >= 0) {
            uint64_t ep_bb = sq_bb(b->ep_square);
            uint64_t ep_left  = (pawns >> 7) & ~FILE_A & ep_bb;
            uint64_t ep_right = (pawns >> 9) & ~FILE_H & ep_bb;
            if (ep_left)
                moves[count++] = cmove_make(b->ep_square + 7, b->ep_square, 0,
                                             CMOVE_FLAG_CAPTURE | CMOVE_FLAG_EP);
            if (ep_right)
                moves[count++] = cmove_make(b->ep_square + 9, b->ep_square, 0,
                                             CMOVE_FLAG_CAPTURE | CMOVE_FLAG_EP);
        }
    }

    return count;
}

static int gen_knight_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    uint64_t knights = b->pieces[us][KNIGHT];
    uint64_t own = b->occupied_co[us];
    uint64_t enemies = b->occupied_co[1 - us];

    while (knights) {
        int from = pop_lsb(knights);
        uint64_t targets = KNIGHT_ATTACKS[from] & ~own;
        while (targets) {
            int to = pop_lsb(targets);
            int flags = (enemies & sq_bb(to)) ? CMOVE_FLAG_CAPTURE : 0;
            moves[count++] = cmove_make(from, to, 0, flags);
        }
    }
    return count;
}

static int gen_bishop_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    uint64_t bishops = b->pieces[us][BISHOP];
    uint64_t own = b->occupied_co[us];
    uint64_t enemies = b->occupied_co[1 - us];

    while (bishops) {
        int from = pop_lsb(bishops);
        uint64_t targets = bishop_attacks(from, b->occupied) & ~own;
        while (targets) {
            int to = pop_lsb(targets);
            int flags = (enemies & sq_bb(to)) ? CMOVE_FLAG_CAPTURE : 0;
            moves[count++] = cmove_make(from, to, 0, flags);
        }
    }
    return count;
}

static int gen_rook_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    uint64_t rooks = b->pieces[us][ROOK];
    uint64_t own = b->occupied_co[us];
    uint64_t enemies = b->occupied_co[1 - us];

    while (rooks) {
        int from = pop_lsb(rooks);
        uint64_t targets = rook_attacks(from, b->occupied) & ~own;
        while (targets) {
            int to = pop_lsb(targets);
            int flags = (enemies & sq_bb(to)) ? CMOVE_FLAG_CAPTURE : 0;
            moves[count++] = cmove_make(from, to, 0, flags);
        }
    }
    return count;
}

static int gen_queen_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    uint64_t queens = b->pieces[us][QUEEN];
    uint64_t own = b->occupied_co[us];
    uint64_t enemies = b->occupied_co[1 - us];

    while (queens) {
        int from = pop_lsb(queens);
        uint64_t targets = queen_attacks(from, b->occupied) & ~own;
        while (targets) {
            int to = pop_lsb(targets);
            int flags = (enemies & sq_bb(to)) ? CMOVE_FLAG_CAPTURE : 0;
            moves[count++] = cmove_make(from, to, 0, flags);
        }
    }
    return count;
}

static int gen_king_moves(const struct CBoard* b, CMove* moves) {
    int count = 0;
    int us = b->side_to_move;
    uint64_t king_bb = b->pieces[us][KING];
    if (!king_bb) return 0;

    int from = lsb(king_bb);
    uint64_t own = b->occupied_co[us];
    uint64_t enemies = b->occupied_co[1 - us];

    // Normal king moves
    uint64_t targets = KING_ATTACKS[from] & ~own;
    while (targets) {
        int to = pop_lsb(targets);
        int flags = (enemies & sq_bb(to)) ? CMOVE_FLAG_CAPTURE : 0;
        moves[count++] = cmove_make(from, to, 0, flags);
    }

    // Castling
    int them = 1 - us;
    if (us == 0) { // White
        // Kingside: e1-g1 (squares f1, g1 must be empty and not attacked; e1 not in check)
        if ((b->castling & 1) && from == 4) {
            if (!(b->occupied & (sq_bb(5) | sq_bb(6))) &&
                !is_attacked_by(b, 4, them) &&
                !is_attacked_by(b, 5, them) &&
                !is_attacked_by(b, 6, them)) {
                moves[count++] = cmove_make(4, 6, 0, CMOVE_FLAG_CASTLE);
            }
        }
        // Queenside: e1-c1 (squares b1, c1, d1 must be empty; c1, d1, e1 not attacked)
        if ((b->castling & 2) && from == 4) {
            if (!(b->occupied & (sq_bb(1) | sq_bb(2) | sq_bb(3))) &&
                !is_attacked_by(b, 4, them) &&
                !is_attacked_by(b, 3, them) &&
                !is_attacked_by(b, 2, them)) {
                moves[count++] = cmove_make(4, 2, 0, CMOVE_FLAG_CASTLE);
            }
        }
    } else { // Black
        // Kingside: e8-g8
        if ((b->castling & 4) && from == 60) {
            if (!(b->occupied & (sq_bb(61) | sq_bb(62))) &&
                !is_attacked_by(b, 60, them) &&
                !is_attacked_by(b, 61, them) &&
                !is_attacked_by(b, 62, them)) {
                moves[count++] = cmove_make(60, 62, 0, CMOVE_FLAG_CASTLE);
            }
        }
        // Queenside: e8-c8
        if ((b->castling & 8) && from == 60) {
            if (!(b->occupied & (sq_bb(57) | sq_bb(58) | sq_bb(59))) &&
                !is_attacked_by(b, 60, them) &&
                !is_attacked_by(b, 59, them) &&
                !is_attacked_by(b, 58, them)) {
                moves[count++] = cmove_make(60, 58, 0, CMOVE_FLAG_CASTLE);
            }
        }
    }

    return count;
}

// ─── Legality check: make move on temp board and verify king not in check ───
// Returns true if the position after the move leaves our king safe.
static bool is_legal(const struct CBoard* b, CMove mv) {
    int us = b->side_to_move;
    int them = 1 - us;
    int from = CMOVE_FROM(mv);
    int to = CMOVE_TO(mv);
    int flags = CMOVE_FLAGS(mv);

    // For castling, we already checked attacks in gen_king_moves
    if (flags & CMOVE_FLAG_CASTLE)
        return true;

    // Make a copy for simulation
    struct CBoard temp;
    memcpy(&temp, b, sizeof(struct CBoard));

    // Remove piece from 'from'
    int piece_type = 0;
    for (int pt = PAWN; pt <= KING; pt++) {
        if (temp.pieces[us][pt] & sq_bb(from)) {
            piece_type = pt;
            temp.pieces[us][pt] &= ~sq_bb(from);
            break;
        }
    }

    // Handle captures
    if (flags & CMOVE_FLAG_EP) {
        // En passant: remove the pawn behind the target square
        int cap_sq = (us == 0) ? to - 8 : to + 8;
        temp.pieces[them][PAWN] &= ~sq_bb(cap_sq);
        temp.occupied_co[them] &= ~sq_bb(cap_sq);
    } else if (flags & CMOVE_FLAG_CAPTURE) {
        // Remove captured piece
        for (int pt = PAWN; pt <= KING; pt++) {
            if (temp.pieces[them][pt] & sq_bb(to)) {
                temp.pieces[them][pt] &= ~sq_bb(to);
                break;
            }
        }
        temp.occupied_co[them] &= ~sq_bb(to);
    }

    // Place piece at 'to'
    int placed_type = piece_type;
    if (flags & CMOVE_FLAG_PROMOTION) {
        int promo = CMOVE_PROMO(mv);
        // promo: 1=knight, 2=bishop, 3=rook, 4=queen
        static const int promo_map[] = {0, KNIGHT, BISHOP, ROOK, QUEEN};
        placed_type = promo_map[promo];
    }
    temp.pieces[us][placed_type] |= sq_bb(to);

    // Update occupied bitboards
    temp.occupied_co[us] = (temp.occupied_co[us] & ~sq_bb(from)) | sq_bb(to);
    temp.occupied = temp.occupied_co[0] | temp.occupied_co[1];

    // Check if our king is in check
    uint64_t king_bb = temp.pieces[us][KING];
    if (!king_bb) return false;
    int king_sq = lsb(king_bb);

    return !is_attacked_by(&temp, king_sq, them);
}

// ─── Legal move generation ───
MOVEGEN_API int cboard_legal_moves(const struct CBoard* b, CMove* moves) {
    movegen_init();

    CMove pseudo[256];
    int pseudo_count = 0;

    pseudo_count += gen_pawn_moves(b, pseudo + pseudo_count);
    pseudo_count += gen_knight_moves(b, pseudo + pseudo_count);
    pseudo_count += gen_bishop_moves(b, pseudo + pseudo_count);
    pseudo_count += gen_rook_moves(b, pseudo + pseudo_count);
    pseudo_count += gen_queen_moves(b, pseudo + pseudo_count);
    pseudo_count += gen_king_moves(b, pseudo + pseudo_count);

    // Filter for legality
    int legal_count = 0;
    for (int i = 0; i < pseudo_count; i++) {
        if (is_legal(b, pseudo[i])) {
            moves[legal_count++] = pseudo[i];
        }
    }

    return legal_count;
}
