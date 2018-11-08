#ifndef _FIRST_ROW_INC_
#define _FIRST_ROW_INC_

#define FIRST_ROW__abcdef(VERSION)\
  FLAG_##VERSION##_FIRST_ROW__abcdef:\
  if (condition_x1) {\
    BLOCK_x = NEW_LABEL;\
    LOOP_HANDLING\
    goto FLAG_##VERSION##_FIRST_ROW__abcd;\
  } else {\
    if (condition_x2) {\
      BLOCK_x = NEW_LABEL;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW_e_abcd;\
    } else {\
      if (condition_x3) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcde;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW_f_abcde;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    }\
  }\


#define FIRST_ROW_f_abcde(VERSION)\
  FLAG_##VERSION##_FIRST_ROW_f_abcde:\
  if (condition_x1) {\
    BLOCK_x = BLOCK_f;\
    LOOP_HANDLING\
    goto FLAG_##VERSION##_FIRST_ROW__abcd;\
  } else {\
    if (condition_x3) {\
      BLOCK_x = BLOCK_f;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcd;\
    } else {\
      if (condition_x2) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW_e_abcd;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW_f_abcde;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    }\
  }\


#define FIRST_ROW__abcd(VERSION)\
  FLAG_##VERSION##_FIRST_ROW__abcd:\
  if (condition_x1) {\
    if (condition_e) {\
      BLOCK_x = BLOCK_e;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcd;\
    } else {\
      if (condition_f) {\
        BLOCK_x = BLOCK_f;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcd;\
      } else {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcd;\
      }\
    }\
  } else {\
    if (condition_x3) {\
      if (condition_e) {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcd;\
      } else {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcd;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcd;\
        }\
      }\
    } else {\
      if (condition_x2) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW_e_abcd;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW_f_abcde;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    }\
  }\


#define FIRST_ROW_e_abcd(VERSION)\
  FLAG_##VERSION##_FIRST_ROW_e_abcd:\
  if (condition_x1) {\
    BLOCK_x = BLOCK_e;\
    LOOP_HANDLING\
    goto FLAG_##VERSION##_FIRST_ROW__abcd;\
  } else {\
    if (condition_x3) {\
      BLOCK_x = BLOCK_e;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcd;\
    } else {\
      if (condition_x2) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW_e_abcd;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW_f_abcde;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    }\
  }\


#define FIRST_ROW__abcde(VERSION)\
  FLAG_##VERSION##_FIRST_ROW__abcde:\
  if (condition_x1) {\
    if (condition_f) {\
      BLOCK_x = BLOCK_f;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcd;\
    } else {\
      BLOCK_x = NEW_LABEL;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcd;\
    }\
  } else {\
    if (condition_x3) {\
      if (condition_f) {\
        BLOCK_x = BLOCK_f;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcd;\
      } else {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcd;\
      }\
    } else {\
      if (condition_x2) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW_e_abcd;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW_f_abcde;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    }\
  }\


#define FIRST_ROW_EVEN(VERSION)\
  FIRST_ROW__abcdef(VERSION)\
  FIRST_ROW_f_abcde(VERSION)\
  FIRST_ROW_e_abcd(VERSION)\
  FIRST_ROW__abcde(VERSION)\
  END_FIRST_ROW_##VERSION :\
  FIRST_ROW__abcd(VERSION)\
  FIRST_ROW_UNUSED(VERSION)


#define FIRST_ROW__x2x4abcd(VERSION)\
  FLAG_##VERSION##_FIRST_ROW__x2x4abcd:\
  if (condition_x1) {\
    if (condition_e) {\
      BLOCK_x = BLOCK_e;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
    } else {\
      if (condition_f) {\
        BLOCK_x = BLOCK_f;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
      } else {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
      }\
    }\
  } else {\
    if (condition_x3) {\
      if (condition_e) {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
      } else {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
        }\
      }\
    } else {\
      BLOCK_x = 0;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_FIRST_ROW__abcdef;\
    }\
  }\


#define FIRST_ROW_ODD(VERSION)\
  FIRST_ROW__abcdef(VERSION)\
  FIRST_ROW_f_abcde(VERSION)\
  FIRST_ROW_e_abcd(VERSION)\
  FIRST_ROW__abcde(VERSION)\
  FIRST_ROW__abcd(VERSION)\
  END_FIRST_ROW_##VERSION :\
  FIRST_ROW__x2x4abcd(VERSION)\
  FIRST_ROW_UNUSED(VERSION)


// Unused predictions for end of rows. To avoid compilation errors.
#define FIRST_ROW_UNUSED(VERSION)\

#endif // _FIRST_ROW_INC_
