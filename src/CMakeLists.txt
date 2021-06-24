target_sources(YACCLAB
    PRIVATE
        check_labeling.cc
        config_data.cc
        file_manager.cc
        labeling_algorithms.cc
        labeling_bolelli_2018.cc
        labeling_bolelli_2019.cc
        labeling_distefano_1999.cc
        labeling_fchang_2003.cc
        labeling_grana_2010.cc
        labeling_grana_2016.cc
        labeling_he_2014.cc
        labeling_he_2008.cc
        labeling_lacassagne_2016.cc
        labeling_null.cc
        labeling_wu_2009.cc
        labeling_wychang_2015.cc
        labeling_zhao_2010.cc
        labels_solver.cc
        main.cc
        system_info.cc
        yacclab_tests.cc
        utilities.cc
        volume_util.cc
        yacclab_tensor.cc
        labeling_sauf_4c.cc
        labeling_sauf_background.cc
		labeling_brts.cc
		labeling_bmrs.cc
)


if(OpenMP_CXX_FOUND) 
    target_sources(YACCLAB
        PRIVATE
        labeling_HT_CCL_Diaz2019_v1_2_0.cc
    )
endif()

if(YACCLAB_ENABLE_3D)
    target_sources(YACCLAB PRIVATE
        labeling3D_naive.cc
        labeling3D_he_2011.cc
        labeling3D_he_2011_run.cc
    )
    if(YACCLAB_ENABLE_EPDT_19C)
        target_sources(YACCLAB PRIVATE
            labeling3D_EPDT_19c.cc
        )
    endif()
    if(YACCLAB_ENABLE_EPDT_22C)
        target_sources(YACCLAB PRIVATE
            labeling3D_EPDT_22c.cc
        )
    endif()
    if(YACCLAB_ENABLE_EPDT_26C)
        target_sources(YACCLAB PRIVATE
            labeling3D_EPDT_26c.cc
        )
    endif()
endif()


