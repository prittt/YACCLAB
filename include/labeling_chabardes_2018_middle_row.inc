#ifndef _MIDDLE_ROW_INC_
#define _MIDDLE_ROW_INC_

#define MIDDLE_ROW__aef(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__aef:\
  if (condition_x2) {\
    if (condition_c) {\
      BLOCK_x = BLOCK_c;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
    } else {\
      if (condition_b) {\
        if (condition_d) {\
          BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
        } else {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
        }\
      } else {\
        if (condition_d) {\
          BLOCK_x = BLOCK_d;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
        }\
      }\
    }\
  } else {\
    if (condition_x1) {\
      if (condition_b) {\
        BLOCK_x = BLOCK_b;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_e;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        }\
      }\
    } else {\
      if (condition_x3) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW__e(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__e:\
  if (condition_x1) {\
    if (condition_a) {\
      if (condition_c) {\
        if (condition_b) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          } else {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          }\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_c, BLOCK_f));\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          } else {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_c);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          }\
        }\
      } else {\
        if (condition_x2) {\
          if (condition_d) {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            }\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              BLOCK_x = BLOCK_a;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          } else {\
            BLOCK_x = BLOCK_a;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          }\
        }\
      }\
    } else {\
      if (condition_c) {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        }\
      } else {\
        if (condition_x2) {\
          if (condition_b) {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = BLOCK_b;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          } else {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = BLOCK_d;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = BLOCK_f;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = NEW_LABEL;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          }\
        } else {\
          if (condition_b) {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
            }\
          } else {\
            if (condition_f) {\
              BLOCK_x = BLOCK_f;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
            }\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          } else {\
            BLOCK_x = BLOCK_c;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          }\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = BLOCK_b;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          } else {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = BLOCK_d;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = BLOCK_f;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = NEW_LABEL;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        }\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_a_(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_a_:\
  if (condition_x1) {\
    if (condition_c) {\
      if (condition_b) {\
        if (condition_e) {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          } else {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          }\
        }\
      } else {\
        if (condition_e) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_c, BLOCK_f));\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          } else {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_c);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
          }\
        }\
      }\
    } else {\
      if (condition_x2) {\
        if (condition_d) {\
          if (condition_e) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            }\
          }\
        } else {\
          if (condition_e) {\
            BLOCK_x = BLOCK_e;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              BLOCK_x = BLOCK_a;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      } else {\
        if (condition_e) {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          } else {\
            BLOCK_x = BLOCK_a;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          if (condition_e) {\
            if (condition_b) {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
            } else {\
              BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
            }\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
            } else {\
              BLOCK_x = BLOCK_c;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
            }\
          }\
        } else {\
          if (condition_d) {\
            if (condition_e) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              if (condition_b) {\
                if (condition_f) {\
                  BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
                } else {\
                  BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
                }\
              } else {\
                if (condition_f) {\
                  BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
                } else {\
                  BLOCK_x = BLOCK_d;\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
                }\
              }\
            }\
          } else {\
            if (condition_e) {\
              BLOCK_x = BLOCK_e;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              if (condition_b) {\
                if (condition_f) {\
                  BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
                } else {\
                  BLOCK_x = BLOCK_b;\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
                }\
              } else {\
                if (condition_f) {\
                  BLOCK_x = BLOCK_f;\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
                } else {\
                  BLOCK_x = NEW_LABEL;\
                  LOOP_HANDLING\
                  goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
                }\
              }\
            }\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        if (condition_e) {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        } else {\
          if (condition_f) {\
            BLOCK_x = BLOCK_f;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__e;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__e;\
          }\
        }\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW__ae(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__ae:\
  if (condition_x2) {\
    if (condition_c) {\
      if (condition_f) {\
        if (condition_x1) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_x3) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          } else {\
            BLOCK_x = BLOCK_c;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          }\
        }\
      } else {\
        BLOCK_x = BLOCK_c;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
      }\
    } else {\
      if (condition_b) {\
        if (condition_d) {\
          if (condition_f) {\
            if (condition_x1) {\
              BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              if (condition_x3) {\
                BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            }\
          } else {\
            BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          }\
        } else {\
          if (condition_f) {\
            if (condition_x1) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              if (condition_x3) {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = BLOCK_b;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          } else {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      } else {\
        if (condition_d) {\
          if (condition_f) {\
            if (condition_x1) {\
              BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              if (condition_x3) {\
                BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = BLOCK_d;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            }\
          } else {\
            BLOCK_x = BLOCK_d;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          }\
        } else {\
          if (condition_f) {\
            if (condition_x1) {\
              BLOCK_x = BLOCK_f;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              if (condition_x3) {\
                BLOCK_x = BLOCK_f;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = NEW_LABEL;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x1) {\
      if (condition_b) {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        } else {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        }\
      } else {\
        if (condition_c) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_e;\
          } else {\
            BLOCK_x = BLOCK_c;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_a_e;\
          }\
        } else {\
          if (condition_f) {\
            BLOCK_x = BLOCK_f;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        }\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_f_e(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_f_e:\
  if (condition_x1) {\
    if (condition_a) {\
      if (condition_c) {\
        if (condition_b) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_c, BLOCK_f));\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        }\
      } else {\
        if (condition_x2) {\
          if (condition_d) {\
            BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        } else {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        }\
      }\
    } else {\
      if (condition_c) {\
        BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
      } else {\
        if (condition_x2) {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_f;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        } else {\
          if (condition_b) {\
            BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          } else {\
            BLOCK_x = BLOCK_f;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_b, MERGE(BLOCK_d, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_f;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        BLOCK_x = BLOCK_f;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_be_a(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_be_a:\
  if (condition_x2) {\
    if (condition_c) {\
      BLOCK_x = BLOCK_b;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
    } else {\
      if (condition_d) {\
        BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
      } else {\
        BLOCK_x = BLOCK_b;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
      }\
    }\
  } else {\
    if (condition_x1) {\
      BLOCK_x = BLOCK_b;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW__e;\
    } else {\
      if (condition_x3) {\
        BLOCK_x = BLOCK_b;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_e_ab(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_e_ab:\
  if (condition_x1) {\
    if (condition_c) {\
      BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
    } else {\
      if (condition_x2) {\
        if (condition_d) {\
          BLOCK_x = MERGE(BLOCK_d, BLOCK_e);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
        } else {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
        }\
      } else {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_d) {\
            BLOCK_x = MERGE(BLOCK_d, BLOCK_e);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = BLOCK_e;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_d) {\
            BLOCK_x = BLOCK_d;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW__ef(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__ef:\
  if (condition_x1) {\
    if (condition_a) {\
      if (condition_c) {\
        if (condition_b) {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_c);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        }\
      } else {\
        if (condition_x2) {\
          if (condition_d) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = BLOCK_a;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        } else {\
          BLOCK_x = BLOCK_a;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        }\
      }\
    } else {\
      if (condition_c) {\
        BLOCK_x = BLOCK_c;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
      } else {\
        if (condition_x2) {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        } else {\
          if (condition_b) {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_c) {\
        BLOCK_x = BLOCK_c;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
      } else {\
        if (condition_b) {\
          if (condition_d) {\
            BLOCK_x = MERGE(BLOCK_b, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        } else {\
          if (condition_d) {\
            BLOCK_x = BLOCK_d;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        BLOCK_x = NEW_LABEL;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_ae_(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_ae_:\
  if (condition_x1) {\
    if (condition_c) {\
      if (condition_b) {\
        BLOCK_x = BLOCK_b;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
      } else {\
        BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
      }\
    } else {\
      if (condition_x2) {\
        if (condition_d) {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
        } else {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
        }\
      } else {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          if (condition_b) {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          } else {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          }\
        } else {\
          if (condition_d) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = BLOCK_e;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__e;\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_a_e(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_a_e:\
  if (condition_x1) {\
    if (condition_c) {\
      if (condition_b) {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        }\
      } else {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_c, BLOCK_f));\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        } else {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_c);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_;\
        }\
      }\
    } else {\
      if (condition_x2) {\
        if (condition_d) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          } else {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
          }\
        } else {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          } else {\
            BLOCK_x = BLOCK_a;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      } else {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        } else {\
          BLOCK_x = BLOCK_a;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ae;\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_c) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          } else {\
            BLOCK_x = BLOCK_c;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
          }\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_d, BLOCK_f));\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = BLOCK_b;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          } else {\
            if (condition_d) {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_d, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              } else {\
                BLOCK_x = BLOCK_d;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
              }\
            } else {\
              if (condition_f) {\
                BLOCK_x = BLOCK_f;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = NEW_LABEL;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          }\
        }\
      } else {\
        if (condition_c) {\
          BLOCK_x = BLOCK_c;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_ae_;\
        } else {\
          if (condition_b) {\
            if (condition_d) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_d);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          } else {\
            if (condition_d) {\
              BLOCK_x = BLOCK_d;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_be_a;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            }\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__e;\
        }\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_e;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW__d(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__d:\
  if (condition_x1) {\
    if (condition_b) {\
      if (condition_e) {\
        BLOCK_x = BLOCK_b;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__b;\
      } else {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__b;\
        } else {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__b;\
        }\
      }\
    } else {\
      if (condition_c) {\
        if (condition_e) {\
          BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_a_b;\
        } else {\
          if (condition_a) {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, MERGE(BLOCK_c, BLOCK_f));\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_a_b;\
            } else {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_c);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_a_b;\
            }\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_a_b;\
            } else {\
              BLOCK_x = BLOCK_c;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_a_b;\
            }\
          }\
        }\
      } else {\
        if (condition_e) {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__ab;\
        } else {\
          if (condition_a) {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ab;\
            } else {\
              BLOCK_x = BLOCK_a;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ab;\
            }\
          } else {\
            if (condition_f) {\
              BLOCK_x = BLOCK_f;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ab;\
            } else {\
              BLOCK_x = NEW_LABEL;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW__ab;\
            }\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x2) {\
      if (condition_x3) {\
        if (condition_b) {\
          if (condition_e) {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_b;\
          } else {\
            if (condition_f) {\
              BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_b;\
            } else {\
              BLOCK_x = BLOCK_b;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_b;\
            }\
          }\
        } else {\
          if (condition_c) {\
            if (condition_e) {\
              BLOCK_x = MERGE(BLOCK_c, BLOCK_e);\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_ae_b;\
            } else {\
              if (condition_f) {\
                BLOCK_x = MERGE(BLOCK_c, BLOCK_f);\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_ae_b;\
              } else {\
                BLOCK_x = BLOCK_c;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_ae_b;\
              }\
            }\
          } else {\
            if (condition_e) {\
              BLOCK_x = BLOCK_e;\
              LOOP_HANDLING\
              goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
            } else {\
              if (condition_f) {\
                BLOCK_x = BLOCK_f;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              } else {\
                BLOCK_x = NEW_LABEL;\
                LOOP_HANDLING\
                goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
              }\
            }\
          }\
        }\
      } else {\
        if (condition_b) {\
          BLOCK_x = BLOCK_b;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_e_b;\
        } else {\
          if (condition_c) {\
            BLOCK_x = BLOCK_c;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_ae_b;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW_e_ab;\
          }\
        }\
      }\
    } else {\
      if (condition_x3) {\
        if (condition_e) {\
          BLOCK_x = BLOCK_e;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__be;\
        } else {\
          if (condition_f) {\
            BLOCK_x = BLOCK_f;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__be;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__be;\
          }\
        }\
      } else {\
        if (condition_x4) {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW_f_be;\
        } else {\
          BLOCK_x = 0;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__bef;\
        }\
      }\
    }\
  }\


#define MIDDLE_ROW_EVEN(VERSION)\
  MIDDLE_ROW__aef(VERSION)\
  MIDDLE_ROW__e(VERSION)\
  MIDDLE_ROW_a_(VERSION)\
  MIDDLE_ROW__ae(VERSION)\
  MIDDLE_ROW_f_e(VERSION)\
  MIDDLE_ROW_be_a(VERSION)\
  MIDDLE_ROW_e_ab(VERSION)\
  MIDDLE_ROW__ef(VERSION)\
  MIDDLE_ROW_ae_(VERSION)\
  MIDDLE_ROW_a_e(VERSION)\
  END_MIDDLE_ROW_##VERSION :\
  MIDDLE_ROW__d(VERSION)\
  MIDDLE_ROW_UNUSED(VERSION)


#define MIDDLE_ROW__x2x4cd(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__x2x4cd:\
  if (condition_x1) {\
    if (condition_e) {\
      BLOCK_x = BLOCK_e;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
    } else {\
      if (condition_a) {\
        if (condition_f) {\
          BLOCK_x = MERGE(BLOCK_a, BLOCK_f);\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
        } else {\
          BLOCK_x = BLOCK_a;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
        }\
      } else {\
        if (condition_b) {\
          if (condition_f) {\
            BLOCK_x = MERGE(BLOCK_b, BLOCK_f);\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
          } else {\
            BLOCK_x = BLOCK_b;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
          }\
        } else {\
          if (condition_f) {\
            BLOCK_x = BLOCK_f;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
          } else {\
            BLOCK_x = NEW_LABEL;\
            LOOP_HANDLING\
            goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
          }\
        }\
      }\
    }\
  } else {\
    if (condition_x3) {\
      if (condition_e) {\
        BLOCK_x = BLOCK_e;\
        LOOP_HANDLING\
        goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
      } else {\
        if (condition_f) {\
          BLOCK_x = BLOCK_f;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
        } else {\
          BLOCK_x = NEW_LABEL;\
          LOOP_HANDLING\
          goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
        }\
      }\
    } else {\
      BLOCK_x = 0;\
      LOOP_HANDLING\
      goto FLAG_##VERSION##_MIDDLE_ROW__abef;\
    }\
  }\


#define MIDDLE_ROW_ODD(VERSION)\
  MIDDLE_ROW__aef(VERSION)\
  MIDDLE_ROW__e(VERSION)\
  MIDDLE_ROW_a_(VERSION)\
  MIDDLE_ROW__ae(VERSION)\
  MIDDLE_ROW_f_e(VERSION)\
  MIDDLE_ROW_be_a(VERSION)\
  MIDDLE_ROW_e_ab(VERSION)\
  MIDDLE_ROW__ef(VERSION)\
  MIDDLE_ROW_ae_(VERSION)\
  MIDDLE_ROW_a_e(VERSION)\
  END_MIDDLE_ROW_##VERSION :\
  MIDDLE_ROW__x2x4cd(VERSION)\
  MIDDLE_ROW_UNUSED(VERSION)


// Unused predictions for end of rows. To avoid compilation errors.
#define MIDDLE_ROW__ab(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__ab:;
#define MIDDLE_ROW_f_be(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_f_be:;
#define MIDDLE_ROW__abef(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__abef:;
#define MIDDLE_ROW_a_b(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_a_b:;
#define MIDDLE_ROW__b(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__b:;
#define MIDDLE_ROW_ae_b(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_ae_b:;
#define MIDDLE_ROW_e_b(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW_e_b:;
#define MIDDLE_ROW__bef(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__bef:;
#define MIDDLE_ROW__be(VERSION)\
  FLAG_##VERSION##_MIDDLE_ROW__be:;
#define MIDDLE_ROW_UNUSED(VERSION)\
  MIDDLE_ROW__ab(VERSION)\
  MIDDLE_ROW_f_be(VERSION)\
  MIDDLE_ROW__abef(VERSION)\
  MIDDLE_ROW_a_b(VERSION)\
  MIDDLE_ROW__b(VERSION)\
  MIDDLE_ROW_ae_b(VERSION)\
  MIDDLE_ROW_e_b(VERSION)\
  MIDDLE_ROW__bef(VERSION)\
  MIDDLE_ROW__be(VERSION)\

#endif // _MIDDLE_ROW_INC_
