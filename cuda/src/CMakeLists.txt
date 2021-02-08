target_sources(YACCLAB  PRIVATE
    gpu_mat3.cu
    labeling_8DLS.cu
    labeling_allegretti_2019_BBDT.cu
    labeling_allegretti_2019_BKE.cu
    labeling_allegretti_2019_BUF.cu
    labeling_allegretti_2019_DRAG.cu
    labeling_allegretti_2019_SAUF.cu
    labeling_BE_big_tex.cu
    labeling_BKE_InlineCompression.cu
    labeling_BKE_NoInlineCompression.cu
    labeling_BUF_2S.cu
    labeling_BUF_NoInlineCompression.cu
    labeling_cabaret_2017.cu
    labeling_chen_2011.cu
    labeling_chen_2011_1024.cu
    labeling_chen_2011_2048.cu
    labeling_chen_2011_512.cu
    labeling_kalentev_2011.cu
    labeling_KE.cu
    labeling_KE4.cu
    labeling_KE_2S.cu
    labeling_LBUF.cu
    labeling_M8DLS.cu
    labeling_OLE_tex.cu
    labeling_oliveira_2010.cu
    labeling_RADAR.cu
    labeling_rasmusson_2013.cu
    labeling_stava_2011.cu
    labeling_TUF.cu
    labeling_UF_InlineCompression.cu
    labeling_UF_naive.cu
    labeling_zavalishin_2016.cu

)

if(YACCLAB_ENABLE_3D)
    target_sources(YACCLAB PRIVATE        
        labeling_UF_3D.cu
        labeling_BE_3D_orig.cu
        labeling_BKE_3D.cu
        labeling_BUF_3D.cu
        labeling_BUF_3D_NoInlineCompression.cu
        labeling_BE_3D_light.cu
        labeling_BE_3D_light_shared.cu
    )
endif()