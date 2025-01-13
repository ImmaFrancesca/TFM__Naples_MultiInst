import copy
import sys

import cv2
import matplotlib
from pySPICElib.kernelFetch import kernelFetch
from pySPICElib.SPICEtools import *
import spiceypy as spice
from FuturePackage import Instrument
from FuturePackage import ROIDataBase
from FuturePackage import DataManager
from FuturePackage.oplanClassMulti import oplan
#from plotSchedule import plotSchedule
#from plotGanntSchedule import plotGanntSchedule
from genetic.ooamaga import amaga
import os
import pickle
import os
import pandas as pd
#################################################################################################################


#################################################################################################################
target_body = "GANYMEDE"  # Can be a list of strings or a single string

if target_body == "GANYMEDE":
    METAKR = ['https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_comms_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_conjctn_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_flybys_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_baseline_v03.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_pcw2_s01p00_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_pcw2_s02p00_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_lega_s07p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_s007_01_s06p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_attc_000060_230414_240531_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_attm_000059_240817_240827_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_lpbooms_f160326_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_magboom_f160326_v04.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_majis_scan_zero_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_swi_scan_zero_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_crema_5_1_150lb_23_1_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_crema_5_1_150lb_23_1_baseline_v04.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_ptr_soc_s007_01_s02p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_crema_5_1_150lb_23_1_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_crema_5_1_150lb_23_1_baseline_v04.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_ptr_soc_s007_01_s02p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_v40.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_sci_v17.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_ops_v10.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_dsk_surfaces_v11.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_roi_v02.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_events_crema_5_1_150lb_23_1_v02.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_stations_topo_v01.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/rssd0002.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/earth_topo_050714.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/earthfixediau.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/estrack_v04.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_europa_plasma_torus_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_io_plasma_torus_v05.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_ama_gos_ring_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_halo_ring_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_main_ring_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_the_ring_ext_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_the_gos_ring_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_fixed_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_bus_v07.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_gala_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_janus_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_jmc1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_jmc2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_navcam1_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_navcam2_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb3_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb4_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rwi_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_scm_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mag_v06.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_majis_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mga_apm_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mga_dish_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jdc_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jei_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jeni_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jna_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_nim_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rimemx_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rimepx_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_sapy_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_samy_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str3_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_swi_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_uvs_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_gala_v05.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_janus_v08.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_jmc_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_jmag_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_majis_v08.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_navcam_v01.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_pep_v14.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_radem_v03.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_rime_v04.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_rpwi_v03.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_str_v01.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_swi_v07.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_uvs_v06.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_aux_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/lsk/naif0012.tls',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/pck00011.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/de-403-masses.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/gm_de431.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/inpop19a_moon_pa_v01.bpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/earth_070425_370426_predict.bpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/juice_jup011.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/juice_roi_v01.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/sclk/juice_fict_160326_v02.tsc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_sci_v04.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_struct_v21.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_struct_internal_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_cog_v00.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_cog_000060_230416_240516_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_roi_v02.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/mar085_20200101_20400101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/earthstns_fx_050714.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/estrack_v04.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_earthstns_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup365_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup343_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup344-s2003_j24_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup346_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/de432s.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/inpop19a_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/noe-5-2017-gal-a-reduced_20200101_20380902.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_crema_5_1_150lb_23_1_plan_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_orbc_000060_230414_310721_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_orbm_000059_240817_240827_v01.bsp']

else:
    METAKR = ['https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_b2_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_b2_comms_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_b2_conjctn_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_b2_flybys_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_crema_5_1_150lb_23_1_b2_baseline_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_pcw2_s01p00_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_pcw2_s02p00_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_lega_s07p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_ptr_soc_s007_01_s06p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_attc_000060_230414_240531_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sc_attm_000059_240817_240827_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_lpbooms_f160326_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_magboom_f160326_v04.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_majis_scan_zero_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_swi_scan_zero_v02.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_crema_5_1_150lb_23_1_b2_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_crema_5_1_150lb_23_1_b2_baseline_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_sa_ptr_soc_s007_01_s02p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_crema_5_1_150lb_23_1_b2_default_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_crema_5_1_150lb_23_1_b2_baseline_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ck/juice_mga_ptr_soc_s007_01_s02p00_v01.bc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_v40.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_sci_v17.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_ops_v11.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_dsk_surfaces_v11.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_roi_v02.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_events_crema_5_1_150lb_23_1_b2_v02.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/juice_stations_topo_v01.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/rssd0002.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/earth_topo_050714.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/earthfixediau.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/estrack_v04.tf',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_europa_plasma_torus_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_io_plasma_torus_v05.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_ama_gos_ring_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_halo_ring_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_main_ring_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_the_ring_ext_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_jup_the_gos_ring_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_fixed_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_bus_v07.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_gala_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_janus_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_jmc1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_jmc2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_navcam1_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_navcam2_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb3_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_lpb4_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rwi_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_scm_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mag_v06.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_majis_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mga_apm_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_mga_dish_v04.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jdc_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jei_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jeni_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_jna_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_pep_nim_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rimemx_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_rimepx_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_sapy_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_samy_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str1_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str2_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_str3_v02.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_swi_v03.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/dsk/juice_sc_uvs_v01.bds',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_gala_v05.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_janus_v08.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_jmc_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_jmag_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_majis_v08.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_navcam_v01.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_pep_v14.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_radem_v03.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_rime_v04.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_rpwi_v03.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_str_v01.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_swi_v07.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_uvs_v06.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/ik/juice_aux_v02.ti',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/lsk/naif0012.tls',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/pck00011.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/de-403-masses.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/gm_de431.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/inpop19a_moon_pa_v01.bpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/earth_070425_370426_predict.bpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/juice_jup011.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/pck/juice_roi_v01.tpc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/sclk/juice_fict_160326_v02.tsc',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_sci_v04.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_struct_v21.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_struct_internal_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_cog_v00.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_cog_000060_230416_240516_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_roi_v02.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/mar085_20200101_20400101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/earthstns_fx_050714.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/estrack_v04.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_earthstns_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup365_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup343_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup344-s2003_j24_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/jup346_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/de432s.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/inpop19a_19900101_20500101.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/noe-5-2017-gal-a-reduced_20200101_20380902.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_crema_5_1_150lb_23_1_b2_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_orbc_000060_230414_310721_v01.bsp',
          'https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_orbm_000059_240817_240827_v01.bsp']

kf = kernelFetch()
kf.ffList(urlKernelL=METAKR, forceDownload=False)

# INPUTS

#   a) ROI INFO
#   a.1) Raw data info
if target_body == 'GANYMEDE':
    ROIs_filename = "../../../data/roi_info/ganymede_roi_info.txt"  # Can be a list of strings or a single string
else:
    ROIs_filename = "../../../data/roi_info/callisto_roi_info.txt"  # Can be a list of strings or a single string

#   a.2) Should you want to create a custom ROI omit/add the above and do:
# customROI = dict()
# customROI['body'] = 'TARGET_BODY'
# customROI['#roi_key'] = 'ROI_NAME/KEY'
# customROI['vertices'] = np.array([['lon0', 'lat0'], ['lon1', 'lat1'], ['lon2', 'lat2'], ['lon3', 'lat3']])

#   a.2) ROIs to be observed (input the roy key)
desiredROIs = []  # To plan for all the ROIs on the raw datafiles either delete de variable or leave it as an empty list

instruments = []
ROIsList = []
#   b) INSTRUMENT AND OBSERVER INFO


observer = 'JUICE'  # Single string (only one observer per schedule)
# INSTRUMENT 1
inst1Type = 'CAMERA'
ifov = 15e-6  # [rad] single 'double' variable
npix = 1735  # single 'int' variable
imageRate = 10  # [ips] single 'int' variable
fs = 20.  # single 'double' variable
instrument = Instrument(inst1Type, ifov, npix, imageRate, fs)
instruments.append(instrument)

#########################################################################################################
# SETUP JANUS & ROIS TO BE CAPTURED
DB = ROIDataBase(ROIs_filename, target_body)
roinames1 = DB.getnames()  # or rois = [customROI1, customROI2...] List of rois as objects of class oPlanRoi. roiDataBase internally creates each instance of the oPlanRois for each desiredRoi
rois = DB.getROIs()
roiL1 = []
obsCov = False
for name in roinames1:
    patron = f"pickle_{name}.cfg"
    for file in os.listdir("../../../data/roi_files"):
        if file == patron:
            with open('../../../data/roi_files/pickle_' + name + '.cfg', "rb") as f:
                  try:
                        s, e, obsET, obsLen, obsImg, obsRes, obsCov = pickle.load(f)
                  except:
                        s, e, obsET, obsLen, obsImg, obsRes = pickle.load(f)
                  tw = stypes.SPICEDOUBLE_CELL(2000)
                  for i in range(len(s)):
                        spice.wninsd(s[i], e[i], tw)
                  for j in range(len(rois)):
                        if rois[j].name == name:
                              rois[j].ROI_InsType = instrument.type
                              if obsCov:
                                    rois[j].initializeObservationDataBase(roitw=tw, timeData=obsLen, nImg=obsImg,
                                                                          res=obsRes, cov=obsCov, mosaic=True)
                                    roiL1.append(rois[j])
                                    continue
                              else:
                                    rois[j].initializeObservationDataBase(roitw=tw, timeData=obsLen, nImg=obsImg,
                                                                          res=obsRes)
                                    roiL1.append(rois[j])
                                    continue
ROIsList.append(roiL1)

# Instrument 2
inst2Type = 'CAMERA'
ifov = 15e-6  # [rad] single 'double' variable
npix = 1735  # single 'int' variable
imageRate = 10  # [ips] single 'int' variable
fs = 20.  # single 'double' variable
instrument = Instrument(inst2Type, ifov, npix, imageRate, fs)
instruments.append(instrument)

DB = ROIDataBase(ROIs_filename, target_body) # ATTENTION: we have to change ROIs_filename if the second camera has different ROIs
roinames2 = DB.getnames()  # or rois = [customROI1, customROI2...] List of rois as objects of class oPlanRoi. roiDataBase internally creates each instance of the oPlanRois for each desiredRoi
rois = DB.getROIs()
roiL2 = []
for name in roinames2:
    patron = f"pickle_{name}.cfg"
    for file in os.listdir("../../../data/roi_files"):
        if file == patron:
            with open('../../../data/roi_files/pickle_' + name + '.cfg', "rb") as f:
                  try:
                        s, e, obsET, obsLen, obsImg, obsRes, obsCov = pickle.load(f)
                  except:
                        s, e, obsET, obsLen, obsImg, obsRes = pickle.load(f)
                  tw = stypes.SPICEDOUBLE_CELL(2000)
                  for i in range(len(s)):
                        spice.wninsd(s[i], e[i], tw)
                  for j in range(len(rois)):
                        if rois[j].name == name:
                              rois[j].ROI_InsType = instrument.type
                              if obsCov:
                                    rois[j].initializeObservationDataBase(roitw=tw, timeData=obsLen, nImg=obsImg,
                                                                          res=obsRes, cov=obsCov, mosaic=True)
                                    roiL2.append(rois[j])
                                    continue
                              else:
                                    rois[j].initializeObservationDataBase(roitw=tw, timeData=obsLen, nImg=obsImg,
                                                                          res=obsRes)
                                    roiL2.append(rois[j])
                                    continue
ROIsList.append(roiL2)

DataManager(ROIsList, instruments, observer)

#np.random.seed(1234)

plan1 = oplan(len(ROIsList))
feasibility = plan1.checkFeasibility()

if not feasibility:
    sys.exit()

#plan1.ranFun()
#plan1.plotGantt()
#print('initial fitness = ', plan1.fitFun())
#plan1.mutFun()
#print("mutated individual's fitness = ", plan1.fitFun())
#plan1.plotGantt()

p= 62
mymaga = amaga(plan1, 500)
mymaga.setOption('nd', int(mymaga.getPopulationSize() * p/100))
mymaga.setOption('ne', int(0.1 * mymaga.getPopulationSize()))
mymaga.setOption('nn', 500-int(mymaga.getPopulationSize() * p/100)-int(0.1 * mymaga.getPopulationSize())-int((mymaga.getPopulationSize() * (0.8 - p/100))))
mymaga.setOption('nm', int((mymaga.getPopulationSize() * (0.8 - p/100))))
mymaga.setOption('nCanMutate', int(0.15 * mymaga.getPopulationSize()))
mymaga.setOption('nCanProcreate', int(0.15 * mymaga.getPopulationSize()))

mymaga.run(300)

filename = "../../../data/mosaics/Ganymede_mosaic.jpg"
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
matplotlib.use('TkAgg')

fig1, ax1 = plt.subplots()
ax1.imshow(img_rgb, extent=[-180, 180, -90, 90])
mymaga.pop[0].plotObservations(ax1, fig1)
homeFolder = os.path.expanduser('~')
dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
if not os.path.isdir(dataFolder):
      os.makedirs(dataFolder, exist_ok=True)
plt.savefig(f'{dataFolder}/CAMERAS')
plt.close()

fig2, ax2 = plt.subplots()
ax2.imshow(img_rgb, extent=[-180, 180, -90, 90])
mymaga.pop[0].plotObservations_2(ax2, fig2)
homeFolder = os.path.expanduser('~')
dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
if not os.path.isdir(dataFolder):
      os.makedirs(dataFolder, exist_ok=True)
plt.savefig(f'{dataFolder}/CAMERA1')
plt.close()

fig3, ax3 = plt.subplots()
ax3.imshow(img_rgb, extent=[-180, 180, -90, 90])
mymaga.pop[0].plotObservations_3(ax3, fig3)
homeFolder = os.path.expanduser('~')
dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
if not os.path.isdir(dataFolder):
      os.makedirs(dataFolder, exist_ok=True)
plt.savefig(f'{dataFolder}/CAMERA2')
plt.close()

#mymaga.plotPopulation2d()

"""
mymaga.plotSatus2d()
plt.title('Multi-Instrument Schedule Optimization', fontweight='bold', fontsize = 18)
plt.xlabel('CAMERA 1 Fitness', fontweight='bold', fontsize = 15)
plt.ylabel('CAMERA 2 Fitness', fontweight='bold', fontsize = 15)
plt.xticks(fontsize = 14, rotation = 45)
plt.yticks(fontsize = 14)
#plt.xlim([0.4, 1.0])
#plt.ylim([1e+7, 5e+8])
plt.grid(True, 'major')
#plt.plot([], [], 'o', color= 'b', label='Pareto Front', markersize=5.0)
#plt.legend()
#plt.savefig(f'same_parameters/paretos/SortByCrowd_BIN_{p}')
plt.show()
"""

mymaga.printStatus()
#mymaga.pop[0].plotGantt()
front_size = mymaga.getFrontSize(0)
print(f'Front Size: {front_size}')


ind1 = copy.deepcopy(mymaga.pop[0])
ind2 = copy.deepcopy(mymaga.pop[1])
roisL = DataManager.getInstance().getROIList()

print('FIRST INDIVIDUAL')
for inst_index in range(ind1.Ninst):
      schedule = []
      nImgInst = ind1.nImgPlan(inst_index)
      if inst_index == 0:
            instrument = 'CAMERA 1'
      else:
            instrument = 'CAMERA 2'

      for roi_index in range(len(roisL[inst_index])):
            observation = {
                  'instrument': instrument,
                  'ROI': roisL[inst_index][roi_index].name,
                  'start': ind1.stol[inst_index][roi_index],
                  'end': None,
                  'obsLen': ind1.obsLen[inst_index][roi_index],
                  'nImg': nImgInst[roi_index],
                  'res': ind1.qroi[inst_index][roi_index],
                  'cov': ind1.croi[inst_index][roi_index]
            }
            schedule.append(copy.deepcopy(observation))
      schedule = sorted(schedule, key=lambda d: d['start'])
      for i in range(len(schedule)):
            schedule[i]['end'] = spice.et2utc(schedule[i]['start'] + schedule[i]['obsLen'], 'C', 0)
            schedule[i]['start'] = spice.et2utc(schedule[i]['start'], 'C', 0)

      df = pd.DataFrame(schedule)

      # Salvataggio del DataFrame in un file CSV
      df.to_excel(f'sample1_{inst_index}.xlsx', index=False)

      print('INSTRUMENT:', instrument)
      for i in range(len(schedule)):
          start = schedule[i]['start']
          end = schedule[i]['end']
          print('ROI: ', schedule[i]['ROI'],'from: ', start, 'to', end, 'nImg:', schedule[i]['nImg'],
                'res:',schedule[i]['res'],'cov:', schedule[i]['cov'])
      print('Mean resolution: ', np.mean(ind1.qroi[inst_index]))
      print('Mean coverage: ', np.mean(ind1.croi[inst_index]))

print('SECOND INDIVIDUAL')
for inst_index in range(ind2.Ninst):
      schedule = []
      nImgInst = ind2.nImgPlan(inst_index)
      if inst_index == 0:
            instrument = 'CAMERA 1'
      else:
            instrument = 'CAMERA 2'

      for roi_index in range(len(roisL[inst_index])):
            observation = {
                  'instrument': instrument,
                  'ROI': roisL[inst_index][roi_index].name,
                  'start': ind2.stol[inst_index][roi_index],
                  'end': None,
                  'obsLen': ind2.obsLen[inst_index][roi_index],
                  'nImg': nImgInst[roi_index],
                  'res': ind2.qroi[inst_index][roi_index],
                  'cov': ind2.croi[inst_index][roi_index]
            }
            schedule.append(copy.deepcopy(observation))
      schedule = sorted(schedule, key=lambda d: d['start'])
      for i in range(len(schedule)):
            schedule[i]['end'] = spice.et2utc(schedule[i]['start'] + schedule[i]['obsLen'], 'C', 0)
            schedule[i]['start'] = spice.et2utc(schedule[i]['start'], 'C', 0)
      # Creazione del DataFrame
      df = pd.DataFrame(schedule)

      # Salvataggio del DataFrame in un file CSV
      df.to_excel(f'sample2_{inst_index}.xlsx', index=False)
      print('INSTRUMENT:', instrument)
      for i in range(len(schedule)):
          start = schedule[i]['start']
          end = schedule[i]['end']
          print('ROI: ', schedule[i]['ROI'],'from: ', start, 'to', end, 'nImg:', schedule[i]['nImg'],
                'res:',schedule[i]['res'],'cov:', schedule[i]['cov'])
      print('Mean resolution: ', np.mean(ind2.qroi[inst_index]))
      print('Mean coverage: ', np.mean(ind2.croi[inst_index]))


print('FIRST INDIVIDUAL')
schedule = []
for inst_index in range(ind1.Ninst):

      nImgInst = ind1.nImgPlan(inst_index)
      if inst_index == 0:
            instrument = 'CAMERA 1'
      else:
            instrument = 'CAMERA 2'

      for roi_index in range(len(roisL[inst_index])):
            observation = {
                  'instrument': instrument,
                  'ROI': roisL[inst_index][roi_index].name,
                  'start': ind1.stol[inst_index][roi_index],
                  'end': None,
                  'obsLen': ind1.obsLen[inst_index][roi_index],
                  'nImg': nImgInst[roi_index],
                  'res': ind1.qroi[inst_index][roi_index],
                  'cov': ind1.croi[inst_index][roi_index]
            }
            schedule.append(copy.deepcopy(observation))
schedule = sorted(schedule, key=lambda d: d['start'])
for i in range(len(schedule)):
      schedule[i]['end'] = spice.et2utc(schedule[i]['start'] + schedule[i]['obsLen'], 'C', 0)
      schedule[i]['start'] = spice.et2utc(schedule[i]['start'], 'C', 0)

df = pd.DataFrame(schedule)

# Salvataggio del DataFrame in un file CSV
df.to_excel(f'sample1.xlsx', index=False)

print('SECOND INDIVIDUAL')
schedule = []
for inst_index in range(ind2.Ninst):

      nImgInst = ind2.nImgPlan(inst_index)
      if inst_index == 0:
            instrument = 'CAMERA 1'
      else:
            instrument = 'CAMERA 2'

      for roi_index in range(len(roisL[inst_index])):
            observation = {
                  'instrument': instrument,
                  'ROI': roisL[inst_index][roi_index].name,
                  'start': ind2.stol[inst_index][roi_index],
                  'end': None,
                  'obsLen': ind2.obsLen[inst_index][roi_index],
                  'nImg': nImgInst[roi_index],
                  'res': ind2.qroi[inst_index][roi_index],
                  'cov': ind2.croi[inst_index][roi_index]
            }
            schedule.append(copy.deepcopy(observation))
schedule = sorted(schedule, key=lambda d: d['start'])
for i in range(len(schedule)):
      schedule[i]['end'] = spice.et2utc(schedule[i]['start'] + schedule[i]['obsLen'], 'C', 0)
      schedule[i]['start'] = spice.et2utc(schedule[i]['start'], 'C', 0)

df = pd.DataFrame(schedule)

# Salvataggio del DataFrame in un file CSV
df.to_excel(f'sample2.xlsx', index=False)