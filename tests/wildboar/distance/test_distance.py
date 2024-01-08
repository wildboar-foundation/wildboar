# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from wildboar.datasets import load_gun_point, load_two_lead_ecg
from wildboar.distance import (
    argmin_distance,
    argmin_subsequence_distance,
    distance_profile,
    paired_distance,
    paired_subsequence_distance,
    pairwise_distance,
    pairwise_subsequence_distance,
    subsequence_match,
)
from wildboar.distance._distance import _METRICS, _SUBSEQUENCE_METRICS


def _test_metric(a, b):
    import numpy as np

    return np.linalg.norm(a - b)


_EXPECTED_PAIRWISE_DISTANCE = {
    "lcss": {
        None: [
            [0.012195121951219523, 0.03658536585365857, 0.04878048780487809],
            [0.0, 0.024390243902439046, 0.024390243902439046],
            [0.0, 0.03658536585365857, 0.024390243902439046],
        ]
    },
    "edr": {
        None: [
            [0.5975609756097561, 0.47560975609756095, 0.6463414634146342],
            [0.08536585365853659, 0.036585365853658534, 0.13414634146341464],
            [0.0975609756097561, 0.25609756097560976, 0.12195121951219512],
        ]
    },
    "erp": {
        None: [
            [35.701564208604395, 23.691653557121754, 38.59651095978916],
            [10.988020450808108, 4.921718027442694, 13.570069573819637],
            [10.255016681738198, 19.7473459597677, 9.590815722942352],
        ]
    },
    "msm": {
        None: [
            [35.57120902929455, 28.885248728096485, 40.931326208636165],
            [11.86974082980305, 12.198534050490707, 17.675189964473248],
            [10.293032762594521, 23.823819940909743, 14.209565471857786],
        ]
    },
    "twe": {
        None: [
            [68.35703983290492, 51.94744984030726, 75.37315617892149],
            [21.540885984838013, 17.268663048364207, 30.4370250165165],
            [19.225670489326124, 38.6009835104645, 22.736767564475553],
        ]
    },
    "dtw": {
        None: [
            [3.5537758584340273, 2.4239916042161553, 4.014102144666234],
            [1.1052711252130065, 0.6155502223457359, 1.3603006745123651],
            [0.8904511837689495, 1.7621925843838169, 0.9011076709882901],
        ]
    },
    "wdtw": {
        None: [
            [1.2552015453334, 0.840657845575087, 1.397156676128907],
            [0.38125602031080913, 0.21652193787948482, 0.4816084512219569],
            [0.3074476369386968, 0.6205562983854465, 0.32328277683306095],
        ]
    },
    "ddtw": {
        None: [
            [0.8813726884722624, 0.845112685537995, 0.900431284098585],
            [0.5410051293098898, 0.2817966461521348, 0.5468055721838778],
            [0.48507795268129195, 0.5978382928900522, 0.6375355302587586],
        ]
    },
    "wddtw": {
        None: [
            [0.31308613461101015, 0.31035907083813735, 0.32896305445930935],
            [0.1899586497011622, 0.10291886213457994, 0.20003786125651232],
            [0.17004678165807582, 0.22040991627513995, 0.2315062579605518],
        ]
    },
}


_EXPECTED_PAIRWISE_ELASTIC_DISTANCE = {
    "lcss": {
        None: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    },
    "erp": {
        None: [
            [11.895995151251554, 17.091679483652115, 7.37759099714458],
            [12.14670612406917, 17.1011259064544, 12.251492985291407],
            [10.75756766833365, 11.364691082388163, 17.637203577905893],
        ]
    },
    "msm": {
        None: [
            [27.870645819231868, 31.660631390288472, 25.051930798217654],
            [28.350629466818646, 32.191368132131174, 29.716226652963087],
            [26.691123254597187, 26.367045141756535, 32.39279904589057],
        ]
    },
    "twe": {
        None: [
            [36.95036049121615, 44.13520717373491, 30.747075087815496],
            [36.83389494967832, 43.321251590881424, 38.2953548782132],
            [33.929289467781814, 30.94000878250598, 42.822785688996255],
        ]
    },
    "edr": {
        None: [
            [0.07317073170731707, 0.13414634146341464, 0.13414634146341464],
            [0.14634146341463414, 0.17073170731707318, 0.21951219512195122],
            [0.12195121951219512, 0.15853658536585366, 0.3048780487804878],
        ]
    },
    "dtw": {
        None: [
            [1.3184709191755355, 2.2190483689290956, 1.321736156689961],
            [1.792996058064275, 2.6451060425383344, 2.1773549928211353],
            [1.712739739550159, 2.329732794256892, 2.209988047866857],
        ]
    },
    "wdtw": {
        None: [
            [0.6063029568593012, 1.0067804339639945, 0.6091862812454186],
            [0.8658719700761504, 1.2572055560229554, 1.0020888313806977],
            [0.8151014849953684, 1.1236128719385985, 0.9763749677792096],
        ]
    },
    "ddtw": {
        None: [
            [0.3384555598798052, 0.5723072195853035, 0.5743348869276709],
            [0.6519468202845976, 0.6013668447218249, 0.5332778295725764],
            [0.6202644700538644, 0.4273898560390824, 0.37424027871042626],
        ]
    },
    "wddtw": {
        None: [
            [0.15377119394120134, 0.25728089908765234, 0.252656449088879],
            [0.2954656939377251, 0.26678376281035415, 0.229345270589047],
            [0.2704246569028106, 0.17578692671143978, 0.1561584366814303],
        ]
    },
}


_EXPECTED_PAIRWISE_SUBSEQUENCE_DISTANCE = {
    "lcss": {
        None: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    },
    "edr": {
        None: [
            [0.16666666666666666, 0.2, 0.23333333333333334],
            [0.3333333333333333, 0.2, 0.06666666666666667],
            [0.5, 0.16666666666666666, 0.16666666666666666],
            [0.06666666666666667, 0.36666666666666664, 0.43333333333333335],
            [0.5, 0.16666666666666666, 0.26666666666666666],
        ]
    },
    "twe": {
        None: [
            [11.649268791079521, 12.180933441966772, 10.094251576930283],
            [13.415599362820387, 9.74022231733799, 7.110840290904043],
            [19.95431243738532, 9.372541546821594, 8.73246248757839],
            [7.400111306458712, 13.362499579787254, 16.227867022275927],
            [19.025037593103946, 7.749098351947961, 12.236753487698731],
        ]
    },
    "msm": {
        None: [
            [5.884132720530033, 6.212927086278796, 6.066322663798928],
            [7.452256789430976, 5.099573530256748, 4.565707825124264],
            [10.069798408076167, 4.704355984926224, 5.372886322438717],
            [3.7369808349758387, 6.707286648452282, 8.716198019683361],
            [9.660764277447015, 4.539345602039248, 7.1159256673417985],
        ]
    },
    "erp": {
        None: [
            [5.860492743551731, 5.51637732796371, 6.681755607947707],
            [7.235501637682319, 4.641821198165417, 4.494721375405788],
            [9.578443309292197, 4.325426779687405, 5.447692297399044],
            [3.5930565763264894, 6.707286648452282, 8.571289233863354],
            [9.660764277447015, 4.137808752711862, 7.071880591567606],
        ]
    },
    "dtw": {
        None: [
            [0.9892175168657907, 1.0063214093961415, 0.728809497908996],
            [1.1535877966795345, 0.8751305858765588, 0.5918762820384382],
            [1.6463501354826122, 0.8111886944554642, 0.7541254649350714],
            [0.9060083944619378, 1.2025019188465507, 1.0572433054156687],
            [1.6611303630194554, 0.7583611765873354, 0.8452636056135603],
        ]
    },
    "ddtw": {
        None: [
            [0.601288821783055, 0.5221424348860837, 0.3430566541234427],
            [0.5805189571399302, 0.4800321547177683, 0.24825602966276156],
            [0.721921564327584, 0.37348317637184164, 0.5336120107986725],
            [0.7454798658395534, 0.6940890375635499, 0.4371010970146026],
            [0.7298777256758102, 0.4691353842406821, 0.40021046946845795],
        ]
    },
    "wdtw": {
        None: [
            [0.33976588576901373, 0.35696941021281137, 0.29529477646405033],
            [0.40743928135924934, 0.2968451811061202, 0.20069638850403906],
            [0.5718888312347068, 0.27778330485873565, 0.2602774731189233],
            [0.3073448804124781, 0.4393089961212446, 0.43653337575076473],
            [0.5747838846608478, 0.25904902500981475, 0.2936395717131312],
        ]
    },
    "wddtw": {
        None: [
            [0.2101076600612728, 0.18757147836726834, 0.11904901540274042],
            [0.2019113352931962, 0.1691148360424211, 0.08709842622519594],
            [0.25230204368944487, 0.12932437487520157, 0.18590896498560244],
            [0.25993385996374907, 0.24188499844845562, 0.15344920657661343],
            [0.2533634606527797, 0.16450048045815122, 0.1455145521526342],
        ]
    },
    "scaled_dtw": {
        None: [
            [0.7827283778226002, 0.9956500824463717, 0.6349437371068083],
            [0.7097701834905927, 0.8910732527381526, 0.416514036440547],
            [0.816273199927099, 0.7397168194480862, 0.8132343864833252],
            [0.6949114876792972, 0.8384200554809268, 0.9407694709386414],
            [0.7617444669592143, 0.6226157451216432, 0.9970607344464876],
        ]
    },
}

_EXPECTED_SUBSEQUENCE_MATCH = {
    "lcss": {
        None: {
            0.1: (
                [
                    [15, 16, 17, 18, 19, 20, 21, 22],
                    [15, 16, 17, 18, 19, 20, 21],
                    [21, 22, 23, 24, 25, 26, 27, 28, 29],
                ],
                [
                    [
                        0.09999999999999998,
                        0.06666666666666665,
                        0.033333333333333326,
                        0.0,
                        0.0,
                        0.033333333333333326,
                        0.06666666666666665,
                        0.09999999999999998,
                    ],
                    [
                        0.09999999999999998,
                        0.06666666666666665,
                        0.033333333333333326,
                        0.0,
                        0.033333333333333326,
                        0.06666666666666665,
                        0.09999999999999998,
                    ],
                    [
                        0.09999999999999998,
                        0.06666666666666665,
                        0.033333333333333326,
                        0.0,
                        0.0,
                        0.0,
                        0.033333333333333326,
                        0.06666666666666665,
                        0.09999999999999998,
                    ],
                ],
            )
        }
    },
    "edr": {
        None: {
            0.25: (
                [
                    [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                    None,
                    [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                ],
                [
                    [
                        0.23333333,
                        0.2,
                        0.16666667,
                        0.13333333,
                        0.1,
                        0.06666667,
                        0.1,
                        0.13333333,
                        0.16666667,
                        0.2,
                        0.23333333,
                    ],
                    None,
                    [
                        0.23333333,
                        0.2,
                        0.16666667,
                        0.13333333,
                        0.1,
                        0.06666667,
                        0.03333333,
                        0.0,
                        0.03333333,
                        0.06666667,
                        0.1,
                        0.13333333,
                        0.16666667,
                        0.2,
                        0.23333333,
                    ],
                ],
            )
        }
    },
    "adtw": {
        None: {
            1.5: (
                [[17, 18, 19], None, [24, 25, 26]],
                [
                    [1.2761004350294538, 0.700903155292442, 1.174620125782524],
                    None,
                    [1.2007104190288438, 0.6133007904431927, 1.4627049842644184],
                ],
            )
        }
    },
    "dtw": {
        None: {
            1.0: (
                [
                    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                    None,
                    [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                ],
                [
                    [
                        0.69910004,
                        0.55788568,
                        0.55933766,
                        0.54274111,
                        0.55801145,
                        0.54481298,
                        0.56354989,
                        0.64684609,
                        0.67375639,
                        0.96500658,
                    ],
                    None,
                    [
                        0.817363,
                        0.56456229,
                        0.51808363,
                        0.49816442,
                        0.49318993,
                        0.50482461,
                        0.50767,
                        0.51700958,
                        0.5434501,
                        0.66714102,
                    ],
                ],
            )
        }
    },
    "msm": {
        None: {
            10: (
                [
                    [15, 16, 17, 18, 19, 20, 21],
                    [18],
                    [22, 23, 24, 25, 26, 27, 28],
                ],
                [
                    [
                        9.511835888028145,
                        7.39649074152112,
                        5.251493949443102,
                        3.1899966783821583,
                        4.368037987500429,
                        6.296842891722918,
                        8.280407715588808,
                    ],
                    [9.40306032076478],
                    [
                        9.566360974218696,
                        7.281141841318458,
                        5.038193219806999,
                        3.0020722090266645,
                        4.564768569078296,
                        6.472695784177631,
                        8.461842672433704,
                    ],
                ],
            )
        }
    },
    "twe": {
        None: {
            10: (
                [
                    [17, 18, 19, 20],
                    None,
                    [24, 25, 26, 27],
                ],
                [
                    [
                        8.17627472454309,
                        6.233080364763737,
                        7.405248065769669,
                        9.307895302593703,
                    ],
                    None,
                    [
                        7.880466939739877,
                        5.891591119579971,
                        7.1307242352291915,
                        8.987042033366853,
                    ],
                ],
            )
        }
    },
    "erp": {
        None: {
            10: (
                [
                    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                    [18, 19],
                    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                ],
                [
                    [
                        9.182031992822886,
                        7.795043203979731,
                        6.759605066850781,
                        5.8607815355062485,
                        5.3475072011351585,
                        5.041546083986759,
                        4.839753970503807,
                        4.55013020709157,
                        3.878729660063982,
                        3.140085641294718,
                        2.9988022707402706,
                        3.9424597583711147,
                        5.07462539896369,
                        6.127802725881338,
                        7.33895668014884,
                        8.694922473281622,
                        9.919241156429052,
                    ],
                    [9.40306032076478, 9.815009105950594],
                    [
                        8.644489297177643,
                        7.407412046100944,
                        6.272647851612419,
                        5.554487080778927,
                        5.16029088338837,
                        4.889584900345653,
                        4.738964030053467,
                        4.251645693089813,
                        3.5680618821643293,
                        3.0020722090266645,
                        3.443397843744606,
                        4.472180986311287,
                        5.5957204340957105,
                        6.69218667736277,
                        7.8157261847518384,
                        9.006948792841285,
                    ],
                ],
            )
        }
    },
    "wdtw": {
        None: {
            0.3: (
                [
                    [17, 18, 19, 20, 21, 22, 23, 24, 25],
                    None,
                    [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                ],
                [
                    [
                        0.24079630273852187,
                        0.19020777064651523,
                        0.19224316209049055,
                        0.1907945017655901,
                        0.19902953785091396,
                        0.19893070041976205,
                        0.2089376925518019,
                        0.23934627679967999,
                        0.2525752343977616,
                    ],
                    None,
                    [
                        0.28205538471597497,
                        0.1949993697025361,
                        0.17651200562858063,
                        0.16970835502750695,
                        0.17094884854364495,
                        0.1778431154466593,
                        0.18213986259816875,
                        0.1887003235224038,
                        0.20072308574557352,
                        0.24603988739809182,
                    ],
                ],
            )
        }
    },
    "wddtw": {
        None: {
            0.2: (
                [
                    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                    None,
                    [
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                    ],
                ],
                [
                    [
                        0.1809937222255133,
                        0.16753078260958099,
                        0.16577725564878007,
                        0.16003381754414775,
                        0.158055727447161,
                        0.15032548967644252,
                        0.14270999736364304,
                        0.1376070232009899,
                        0.14314838921300463,
                        0.152448147125594,
                        0.16807673676348422,
                        0.18716370253295117,
                    ],
                    None,
                    [
                        0.14816285097215917,
                        0.155616093653265,
                        0.16961960861797967,
                        0.17201756480229888,
                        0.16422484111732608,
                        0.16228619463346677,
                        0.09547664994525815,
                        0.09735729198576938,
                        0.0963088967207191,
                        0.0956271968634347,
                        0.09117268333330547,
                        0.08086078806707415,
                        0.0770191877636208,
                        0.0869757404569166,
                        0.10157044184833051,
                        0.12744267063612733,
                        0.14447502323784212,
                        0.17155479736450815,
                    ],
                ],
            )
        }
    },
    "ddtw": {
        None: {
            0.5: (
                [
                    [7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                    None,
                    [
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                    ],
                ],
                [
                    [
                        0.49182158216783245,
                        0.4628187804407686,
                        0.4328655470944792,
                        0.4375545241480347,
                        0.42895890866552977,
                        0.432847605025534,
                        0.41862230369509923,
                        0.4044146796341606,
                        0.3982195941875438,
                        0.4063916392623821,
                        0.4253976019126424,
                        0.4634947848307937,
                    ],
                    None,
                    [
                        0.4994848540303338,
                        0.354677191176608,
                        0.3902963260973856,
                        0.4427111329212307,
                        0.453419529858458,
                        0.4344569055369136,
                        0.43761726278494323,
                        0.24918812482559735,
                        0.260305918712219,
                        0.26313750035648886,
                        0.266233424332704,
                        0.25597197575796393,
                        0.2293943253247497,
                        0.22293096615846936,
                        0.24771259831923775,
                        0.28573862822244867,
                        0.3571825001345059,
                        0.40059208741587404,
                        0.4791725122034152,
                    ],
                ],
            )
        }
    },
    "scaled_dtw": {
        None: {
            1.0: (
                [
                    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
                    None,
                    [24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                ],
                [
                    [
                        0.8606401222935874,
                        0.6826489565175334,
                        0.6065556826833113,
                        0.5539021010257745,
                        0.6190312467788794,
                        0.7002021618569902,
                        0.718578478149183,
                        0.7589149590400458,
                        0.8646916695428387,
                        0.8717945220644988,
                        0.9710465047353894,
                    ],
                    None,
                    [
                        0.7766221479059982,
                        0.6345536490290729,
                        0.4885936334249242,
                        0.47356241859960985,
                        0.5270622310302566,
                        0.6641269427215154,
                        0.7966532390528641,
                        0.8109244854258868,
                        0.8685013127292864,
                        0.9670775120638503,
                    ],
                ],
            )
        }
    },
}


@pytest.fixture
def X():
    X, _ = load_two_lead_ecg()
    return X


@pytest.fixture
def pairwise_data(X):
    return X[:3], X[3:6]


@pytest.fixture
def pairwise_expected(request):
    metric = request.node.funcargs["metric"]
    metric_params = request.node.funcargs["metric_params"]
    return _EXPECTED_PAIRWISE_DISTANCE[metric][metric_params]


@pytest.fixture
def pairwise_elastic_data(X):
    return X[15:18, 10:-10], X[22:25]


@pytest.fixture
def pairwise_elastic_expected(request):
    metric = request.node.funcargs["metric"]
    metric_params = request.node.funcargs["metric_params"]
    return _EXPECTED_PAIRWISE_ELASTIC_DISTANCE[metric][metric_params]


@pytest.fixture
def pairwise_subsequence_data(X):
    return X[:3, 20:50], X[50:55]


@pytest.fixture
def pairwise_subsequence_distance_expected(request):
    metric = request.node.funcargs["metric"]
    metric_params = request.node.funcargs["metric_params"]
    return _EXPECTED_PAIRWISE_SUBSEQUENCE_DISTANCE[metric][metric_params]


@pytest.fixture
def subsequence_match_data(X):
    return X[3, 20:50], X[52:55]


@pytest.fixture
def subsequence_match_expected(request):
    metric = request.node.funcargs["metric"]
    metric_params = request.node.funcargs["metric_params"]
    threshold = request.node.funcargs["threshold"]
    return _EXPECTED_SUBSEQUENCE_MATCH[metric][metric_params][threshold]


def test_pickle():
    # TODO: ensure that the objects are equivalent
    import pickle

    for Metric in _METRICS.values():
        metric0 = Metric()  # Default params
        p = pickle.dumps(metric0)
        pickle.loads(p)

    for SubsequenceMetric in _SUBSEQUENCE_METRICS.values():
        metric0 = SubsequenceMetric()  # Default params
        p = pickle.dumps(metric0)
        pickle.loads(p)


@pytest.mark.parametrize("metric", list(_METRICS.keys()))
def test_metric_raises_bad_argument(metric):
    bogus_params = {"bogus_param_should_not_exists": 10}
    with pytest.raises(TypeError):
        _METRICS[metric](**bogus_params)


@pytest.mark.parametrize(
    "metric",
    ["dtw", "wdtw", "ddtw", "wddtw", "lcss", "msm", "edr", "erp", "twe"],
)
@pytest.mark.parametrize("metric_params", [{"r": -0.1}, {"r": 1.1}])
def test_raises_bad_r_argument_value(metric, metric_params):
    with pytest.raises(ValueError):
        _METRICS[metric](**metric_params)

    with pytest.raises(ValueError):
        _SUBSEQUENCE_METRICS[metric](**metric_params)


@pytest.mark.parametrize(
    "metric, metric_params",
    [
        ("lcss", {"epsilon": -0.1}),
        ("lcss", {"epsilon": 0}),
        ("msm", {"c": -1}),
        ("edr", {"epsilon": -0.1}),
        ("edr", {"epsilon": 0}),
        ("twe", {"penalty": -0.1}),
        ("twe", {"stiffness": 0.0}),
        ("twe", {"stiffness": -0.1}),
        ("erp", {"g": -0.1}),
    ],
)
def test_raises_bad_argument_value(metric, metric_params):
    with pytest.raises(ValueError):
        _METRICS[metric](**metric_params)

    with pytest.raises(ValueError):
        _SUBSEQUENCE_METRICS[metric](**metric_params)


@pytest.mark.parametrize("metric", list(_SUBSEQUENCE_METRICS.keys()))
def test_subsequence_metric_raises_bad_argument(metric):
    bogus_params = {"bogus_param_should_not_exists": 10}
    with pytest.raises(TypeError):
        _SUBSEQUENCE_METRICS[metric](**bogus_params)


@pytest.mark.parametrize("metric", list(_METRICS.keys()))
def test_pairwise_distance_benchmark(benchmark, metric, X):
    X, y = load_two_lead_ecg()
    x = X[:100].reshape(-1).copy()
    y = X[100:200].reshape(-1).copy()

    benchmark(pairwise_distance, x, y, metric=metric)


@pytest.mark.parametrize("metric", list(_METRICS.keys()))
@pytest.mark.parametrize("k", [1, 3, 5])
def test_argmin_distance_benchmark(benchmark, metric, k):
    X, y = load_two_lead_ecg()
    x = X[:100].copy()
    y = X[100:200].copy()

    benchmark(argmin_distance, x, y, k=k, metric=metric)


@pytest.mark.parametrize("metric", list(_SUBSEQUENCE_METRICS.keys()))
def test_pairwise_subsequence_distance_benchmark(benchmark, metric, X):
    X, y = load_two_lead_ecg()
    x = X[:2].reshape(-1).copy()
    y = X[100:120].reshape(-1).copy()

    benchmark(
        pairwise_subsequence_distance,
        x,
        y,
        metric=metric,
    )


@pytest.mark.parametrize(
    "metric, metric_params",
    [
        ("lcss", None),
        ("erp", None),
        ("edr", None),
        ("msm", None),
        ("twe", None),
        ("dtw", None),
        ("ddtw", None),
        ("wdtw", None),
        ("wddtw", None),
    ],
)
def test_elastic_metric_equal_length(
    metric, metric_params, pairwise_data, pairwise_expected
):
    Y, X = pairwise_data
    actual = pairwise_distance(Y, X, metric=metric, metric_params=metric_params)
    assert_almost_equal(
        actual,
        pairwise_expected,
    )


@pytest.mark.parametrize(
    "metric, metric_params",
    [
        ("lcss", None),
        ("erp", None),
        ("edr", None),
        ("msm", None),
        ("twe", None),
        ("dtw", None),
        ("ddtw", None),
        ("wdtw", None),
        ("wddtw", None),
    ],
)
def test_elastic_metric_unequal_length(
    metric, metric_params, pairwise_elastic_data, pairwise_elastic_expected
):
    Y, X = pairwise_elastic_data
    actual = pairwise_distance(Y, X, metric=metric, metric_params=metric_params)
    assert_almost_equal(
        actual,
        pairwise_elastic_expected,
    )


@pytest.mark.parametrize(
    "metric", ["dtw", "erp", "lcss", "msm", "twe", "ddtw", "wdtw", "wddtw"]
)
def test_pairwise_subsequence_pairwise_equal(metric, pairwise_data):
    Y, X = pairwise_data

    assert_almost_equal(
        pairwise_distance(Y, X, metric=metric),
        pairwise_subsequence_distance(Y, X, metric=metric).T,
    )


@pytest.mark.parametrize(
    "metric, metric_params",
    [
        ("lcss", None),
        ("erp", None),
        ("edr", None),
        ("msm", None),
        ("twe", None),
        ("dtw", None),
        ("ddtw", None),
        ("wdtw", None),
        ("wddtw", None),
        ("scaled_dtw", None),
    ],
)
def test_elastic_pairwise_subsequence_distance(
    metric,
    metric_params,
    pairwise_subsequence_data,
    pairwise_subsequence_distance_expected,
):
    Y, X = pairwise_subsequence_data

    actual = pairwise_subsequence_distance(
        Y, X, metric=metric, metric_params=metric_params
    )
    print(actual.tolist())
    assert_almost_equal(
        actual,
        pairwise_subsequence_distance_expected,
    )


@pytest.mark.parametrize(
    "metric, metric_params, threshold",
    [
        ("lcss", None, 0.1),
        ("erp", None, 10),
        ("edr", None, 0.25),
        ("msm", None, 10),
        ("twe", None, 10),
        ("adtw", None, 1.5),
        ("dtw", None, 1.0),
        ("ddtw", None, 0.5),
        ("wdtw", None, 0.3),
        ("wddtw", None, 0.2),
        ("scaled_dtw", None, 1.0),
    ],
)
def test_subsequence_match(
    metric, metric_params, threshold, subsequence_match_data, subsequence_match_expected
):
    desired_inds, desired_dists = subsequence_match_expected
    Y, X = subsequence_match_data
    actual_inds, actual_dists = subsequence_match(
        Y,
        X,
        metric=metric,
        metric_params=metric_params,
        threshold=threshold,
        return_distance=True,
    )

    assert len(actual_inds) == len(actual_dists)
    for ind, dist in zip(actual_inds, actual_dists):
        if ind is None or dist is None:
            assert ind == dist
        else:
            assert ind.shape == dist.shape
    print(
        [None if i is None else i.tolist() for i in actual_inds],
        [None if i is None else i.tolist() for i in actual_dists],
    )
    for actual_ind, desired_ind in zip(actual_inds, desired_inds):
        if desired_ind is None:
            assert actual_ind is None
        else:
            assert_almost_equal(actual_ind, desired_ind)

    for actual_dist, desired_dist in zip(actual_dists, desired_dists):
        if desired_dist is None:
            assert actual_dist is None
        else:
            assert_almost_equal(actual_dist, desired_dist)


def test_subsequence_match_single():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 3, 3, 3, 1, 2, 3, 4])
    ind = subsequence_match(x, y, threshold=2)
    assert_equal(ind, [0, 1, 6])


def test_subsequence_match_multiple():
    x = np.array([1, 2, 3, 4])
    y = np.array([[1, 2, 3, 3, 3, 3, 1, 2, 3, 4], [10, 2, 3, 3, 3, 3, 1, 2, 3, 4]])
    ind = subsequence_match(x, y, threshold=2)
    desired = np.array([np.array([0, 1, 6]), np.array([1, 6])], dtype=object)
    assert_equal(ind[0], desired[0])
    assert_equal(ind[1], desired[1])


@pytest.mark.parametrize(
    "r, expected",
    [
        pytest.param(
            0.0,
            [
                [
                    9.44862115045952,
                    8.233150190764853,
                    9.293339537019579,
                    8.684918504908113,
                    4.110899010402057,
                ],
                [
                    7.0459917692360285,
                    8.089004876024365,
                    9.693203338390742,
                    8.309887075493558,
                    1.4403655009257892,
                ],
                [
                    6.680197873420652,
                    6.658344132911147,
                    8.019689893011222,
                    7.082917686543071,
                    3.620191236815146,
                ],
            ],
        ),
        pytest.param(
            1.0,
            [
                [
                    2.9831583954146033,
                    4.265599023010943,
                    5.219607373220397,
                    4.487744947315483,
                    0.9254090887540904,
                ],
                [
                    3.1507863403171195,
                    4.438504637974342,
                    5.377807094434376,
                    4.66083452998045,
                    0.5730941885156611,
                ],
                [
                    2.1806556575707536,
                    3.410096002460074,
                    4.3134129556068785,
                    3.6251326980747605,
                    1.8504568621069142,
                ],
            ],
        ),
    ],
)
def test_pairwise_dtw_distance(r, expected):
    x, y = load_gun_point()
    assert_almost_equal(
        pairwise_distance(x[0:3], x[10:15], metric="dtw", metric_params={"r": r}),
        expected,
    )


@pytest.mark.parametrize(
    "r, expected",
    [
        pytest.param(
            0.0,
            [
                10.189318553414651,
                12.135216989056529,
                10.517145480739815,
            ],
            id="r=0.0",
        ),
        pytest.param(
            1.0,
            [
                2.8319246476074187,
                7.559313393620837,
                3.9111923927229193,
            ],
            id="r=1.0",
        ),
    ],
)
def test_paired_dtw_distance(r, expected):
    x, y = load_gun_point()
    assert_almost_equal(
        paired_distance(x[0:3], x[30:33], metric="dtw", metric_params={"r": r}),
        expected,
    )


@pytest.mark.parametrize(
    "metric, metric_params, expected_min_dist, expected_min_ind",
    [
        pytest.param(
            "euclidean",
            None,
            [
                [0.23195688803926817, 0.17188321089000758],
                [0.2674267983857602, 0.15871494464186847],
                [0.3937857247136831, 0.4063928467206117],
                [0.13248433440745758, 0.13046911602986688],
                [1.1448052606769021, 1.2488982882334592],
            ],
            [[108, 116], [21, 22], [106, 106], [114, 113], [2, 2]],
            id="euclidean",
        ),
        pytest.param(
            "scaled_euclidean",
            None,
            [
                [2.7386455797485416, 2.038364843603591],
                [4.358640359464751, 2.809961030159136],
                [3.5622988950840084, 2.9527886868779394],
                [3.032223776553107, 2.571925524686407],
                [3.6662031408030042, 1.914621479718833],
            ],
            [[123, 3], [129, 19], [49, 49], [54, 50], [53, 26]],
            id="scaled_euclidean",
        ),
        pytest.param(
            "dtw",
            {"r": 1.0},
            [
                [0.22001506436908683, 0.1693269004631758],
                [0.25999784868100656, 0.1561472638854919],
                [0.3908331894036726, 0.40533280832526025],
                [0.13213970095179928, 0.12977326096534575],
                [1.1251588562664583, 1.2315231778985767],
            ],
            [[107, 116], [21, 22], [106, 106], [114, 113], [2, 2]],
            id="dtw",
        ),
        pytest.param(
            "scaled_dtw",
            {"r": 1.0},
            [
                [1.3321568849328171, 1.7500375448463599],
                [3.0916513463377413, 1.8913769290869902],
                [1.1870303880302593, 1.7097588566478743],
                [1.7188295859581297, 1.904354845981207],
                [2.854933219112498, 1.6711472926757909],
            ],
            [[126, 3], [23, 21], [52, 48], [54, 51], [128, 27]],
            id="scaled_dtw",
        ),
    ],
)
def test_pairwise_subsequence_distance(
    metric, metric_params, expected_min_dist, expected_min_ind
):
    x, y = load_gun_point()
    min_dist, min_ind = pairwise_subsequence_distance(
        x[[2, 3], 0:20],
        x[40:45],
        metric=metric,
        metric_params=metric_params,
        return_index=True,
    )
    assert_almost_equal(min_dist, expected_min_dist)
    assert_equal(min_ind, expected_min_ind)


@pytest.mark.parametrize(
    "metric, metric_params, expected_min_dist, expected_min_ind",
    [
        pytest.param(
            "euclidean",
            None,
            [0.23195688803926817, 0.15871494464186847, 0.22996935251290676],
            [108, 22, 126],
            id="euclidean",
        ),
        pytest.param(
            "scaled_euclidean",
            None,
            [2.7386455797485416, 2.809961030159136, 2.127943475506045],
            [123, 19, 0],
            id="scaled_euclidean",
        ),
        pytest.param(
            "dtw",
            {"r": 1.0},
            [0.22001506436908683, 0.1561472638854919, 0.16706575311794378],
            [107, 22, 130],
            id="dtw",
        ),
        pytest.param(
            "scaled_dtw",
            {"r": 1.0},
            [1.3321568849328171, 1.8913769290869902, 1.506334150321518],
            [126, 21, 0],
            id="scaled_dtw",
        ),
    ],
)
def test_paired_subsequence_distance(
    metric, metric_params, expected_min_dist, expected_min_ind
):
    x, y = load_gun_point()
    min_dist, min_ind = paired_subsequence_distance(
        x[[2, 3, 8], 0:20],
        x[40:43],
        metric=metric,
        metric_params=metric_params,
        return_index=True,
    )
    assert_almost_equal(min_dist, expected_min_dist)
    assert_equal(min_ind, expected_min_ind)


@pytest.mark.parametrize(
    "metric, metric_params, expected_indicies, expected_dists",
    [
        [
            "euclidean",
            {},
            [
                [126, 127, 125, 128, 129, 130, 124, 131, 132, 123],
                [109, 110, 108, 111, 138, 118, 117, 124, 119, 112],
            ],
            [
                [
                    0.03397965606582041,
                    0.0343842104203779,
                    0.0351323095527136,
                    0.03539658207798578,
                    0.03645583018956124,
                    0.036979988389006124,
                    0.03703322511321164,
                    0.03831106381122556,
                    0.04012265023031157,
                    0.04104258653686285,
                ],
                [
                    0.18783105110608192,
                    0.1962122726590744,
                    0.19969665170517434,
                    0.21358058104153474,
                    0.2286672665170618,
                    0.23024263523833352,
                    0.2310810578541956,
                    0.2323035020498981,
                    0.23286074160686906,
                    0.23352672398649205,
                ],
            ],
        ],
        [
            "scaled_euclidean",
            {},
            [
                [90, 89, 83, 91, 84, 82, 85, 88, 81, 92],
                [75, 89, 88, 87, 71, 90, 86, 76, 70, 72],
            ],
            [
                [
                    0.5171341149025254,
                    0.6106690561241724,
                    0.6199414822706423,
                    0.6389336609057809,
                    0.6861095369454515,
                    0.6919642938064638,
                    0.7518358747794223,
                    0.7551918338623284,
                    0.8089276966112849,
                    0.8305581115853162,
                ],
                [
                    0.5395380605368274,
                    0.5994517516083463,
                    0.6060069282120507,
                    0.6572386395006412,
                    0.6865909258011326,
                    0.7388046565921842,
                    0.7530885210085916,
                    0.810795863777238,
                    0.8376980330763003,
                    0.8501658606924453,
                ],
            ],
        ],
        [
            "dtw",
            {"r": 0.5},
            [
                [126, 127, 125, 128, 129, 130, 124, 131, 132, 123],
                [109, 110, 108, 125, 124, 111, 116, 117, 126, 118],
            ],
            [
                [
                    0.0337779891194657,
                    0.03417138314023305,
                    0.0351323095527136,
                    0.035180236615724184,
                    0.03620168568738906,
                    0.03692724340632057,
                    0.037000014085456216,
                    0.03829773823788508,
                    0.04012265023031157,
                    0.04092858833956632,
                ],
                [
                    0.1859017766089558,
                    0.19554091347043767,
                    0.19601154714547592,
                    0.21089278011500506,
                    0.21265350653754128,
                    0.21347216622866322,
                    0.21404359401413925,
                    0.21451363995569336,
                    0.21813588469124268,
                    0.21833681247621853,
                ],
            ],
        ],
        [
            "scaled_dtw",
            {"r": 0.5},
            [
                [90, 84, 86, 92, 83, 87, 89, 85, 88, 91],
                [75, 72, 71, 89, 90, 88, 74, 87, 91, 77],
            ],
            [
                [
                    0.5171341149025254,
                    0.53438801033894,
                    0.5449097439782348,
                    0.5815724724804437,
                    0.5868331003736715,
                    0.5874821636080592,
                    0.5972173148623663,
                    0.5978139524706985,
                    0.613990263275831,
                    0.6158445792495065,
                ],
                [
                    0.40792177885373476,
                    0.49151144102128486,
                    0.5507076093071009,
                    0.5574072024166825,
                    0.5643688082122642,
                    0.5774972865470762,
                    0.5904850517319787,
                    0.6204986495558397,
                    0.6239597262924556,
                    0.6674760578321446,
                ],
            ],
        ],
    ],
)
def test_subseqence_match_default(
    metric, metric_params, expected_indicies, expected_dists
):
    x, _ = load_gun_point()
    subsequence = x[0, 3:15]
    actual_indices, actual_dists = subsequence_match(
        subsequence,
        x[1:3],
        return_distance=True,
        metric=metric,
        metric_params=metric_params,
    )
    print([c.tolist() for c in actual_dists.tolist()])
    for actual_dist, expected_dist in zip(actual_dists, expected_dists):
        assert_almost_equal(actual_dist, expected_dist)

    for actual_index, expected_index in zip(actual_indices, expected_indicies):
        assert_equal(actual_index, expected_index)


@pytest.mark.parametrize(
    "left, right",
    [
        ["euclidean", "dtw"],
        ["scaled_euclidean", "scaled_dtw"],
        ["scaled_euclidean", "mass"],
        ["mass", "scaled_dtw"],
    ],
)
def test_subsequence_match_equivalent(left, right):
    x, _ = load_gun_point()
    subsequence = x[10, 40:60]

    left_indicies, left_distances = subsequence_match(
        subsequence,
        x[0:9],
        return_distance=True,
        max_matches=20,
        exclude=4,
        metric=left,
        metric_params=dict(r=0.0) if "dtw" in left else {},
    )

    right_indicies, right_distances = subsequence_match(
        subsequence,
        x[0:9],
        return_distance=True,
        max_matches=20,
        exclude=4,
        metric=right,
        metric_params=dict(r=0.0) if "dtw" in right else {},
    )

    for left_distance, right_distance in zip(left_distances, right_distances):
        assert_almost_equal(left_distance, right_distance)

    for left_index, right_index in zip(left_indicies, right_indicies):
        assert_equal(left_index, right_index)


def test_paired_distance_dim_mean():
    X, y = load_two_lead_ecg()
    x = X[:6].reshape(2, 3, -1)
    y = X[6:12].reshape(2, 3, -1)
    expected = np.array([3.8665873284447385, 6.069396564992267])
    actual = paired_distance(x, y, dim="mean", metric="euclidean")
    assert_almost_equal(actual, expected)


def test_paired_distance_dim_full():
    X, y = load_two_lead_ecg()
    x = X[:6].reshape(2, 3, -1)
    y = X[6:12].reshape(2, 3, -1)
    expected = np.array(
        [
            [5.486831916137321, 2.5050700084886945, 3.607860060708199],
            [6.359545576183372, 4.600410679545082, 7.248233439248347],
        ]
    )
    actual = paired_distance(x, y, dim="full", metric="euclidean")
    assert_almost_equal(actual, expected.T)


def test_pairwise_distance_dim_full():
    X, y = load_two_lead_ecg()
    x = X[:6].reshape(2, 3, -1)
    y = X[6:12].reshape(2, 3, -1)
    expected = np.array(
        [
            [
                [5.486831916137321, 6.6030195361586985],
                [4.340837223002717, 6.359545576183372],
            ],
            [
                [2.5050700084886945, 0.9092063468190077],
                [5.276461266463071, 4.600410679545082],
            ],
            [
                [3.607860060708199, 3.756451638398866],
                [6.266771463173544, 7.248233439248347],
            ],
        ]
    )

    actual = pairwise_distance(x, y, dim="full", metric="euclidean")
    assert_almost_equal(actual, expected)


def test_pairwise_distance_dim_mean():
    X, y = load_two_lead_ecg()
    x = X[:6].reshape(2, 3, -1)
    y = X[6:12].reshape(2, 3, -1)

    expected = np.array(
        [
            [3.8665873284447385, 3.7562258404588573],
            [5.294689984213111, 6.069396564992267],
        ]
    )
    actual = pairwise_distance(x, y, dim="mean", metric="euclidean")
    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("metric", _METRICS.keys() | {_test_metric})
@pytest.mark.parametrize("k", [1, 3, 7])
def test_argmin_equals_pairwise_distance_argpartition(metric, k):
    X, y = load_two_lead_ecg()
    X, Y = X[:10], X[300:350]
    ind_argmin, min_dist_argmin = argmin_distance(
        X, Y, metric=metric, k=k, return_distance=True
    )
    ind_argmin = np.sort(ind_argmin, axis=1)

    dist = pairwise_distance(X, Y, metric=metric)
    ind_pairwise = np.argpartition(dist, k, axis=1)[:, :k]
    # ind_pairwise = np.sort(ind_pairwise, axis=1)
    # assert_equal(ind_pairwise, ind_argmin)
    assert_almost_equal(
        np.sort(np.take_along_axis(dist, ind_pairwise, axis=1), axis=1),
        np.sort(min_dist_argmin, axis=1),
    )


@pytest.mark.parametrize(
    "metric", ((_SUBSEQUENCE_METRICS.keys() - {"mass"}) | {_test_metric})
)
@pytest.mark.parametrize("k", [1, 3, 7])
def test_argmin_subsequence_distance(metric, k):
    X, y = load_two_lead_ecg()
    S = np.lib.stride_tricks.sliding_window_view(X[0], window_shape=10)
    X = np.broadcast_to(X[0], shape=(S.shape[0], X.shape[1]))

    metric_params = None

    # The epsilon value by default is computed using the a quarter of the
    # maximum standard deviation of both time series. For the subsequence metrics,
    # its computed as a quarter of the subsequence standard deviation.
    if metric in ("scaled_edr", "edr"):
        metric_params = {"epsilon": 0.1}

    argmin_ind, argmin_dist = argmin_subsequence_distance(
        S,
        X,
        metric=metric,
        metric_params=metric_params,
        k=k,
        return_distance=True,
    )

    for i in range(10):
        dp_dist = distance_profile(
            S[i], X[i], metric=metric, metric_params=metric_params
        )
        dp_ind = np.argpartition(dp_dist, k)[:k]
        assert_almost_equal(np.sort(dp_dist[dp_ind]), np.sort(argmin_dist[i]))


def test_dilated_distance_profile():
    X, y = load_two_lead_ecg()
    dp = distance_profile(
        X[0:2, 9:21], X[2:4], dilation=3, padding=4, metric="euclidean"
    )
    # fmt: off
    expected = np.array([[
        2.11436574, 2.24116065, 2.49773792, 3.27044544, 3.51998601,
        3.6652928 , 4.15694258, 4.29707311, 4.16155973, 4.53797456,
        4.60542518, 4.389539  , 4.68536655, 4.70688945, 4.43782607,
        4.70859085, 4.72284395, 4.43340333, 4.71218903, 4.73662644,
        4.45428471, 4.73445737, 4.74145421, 4.47657337, 4.75591972,
        4.78080953, 4.52822443, 4.81966361, 4.87074441, 4.6246061 ,
        4.91925014, 4.98958115, 4.78785169, 5.12471433, 5.21140376,
        5.01272983, 5.34408054, 5.3696427 , 4.96057778, 4.78043682,
        4.47693369, 3.96784028, 3.7412764 , 3.5475223 , 3.27906092,
        3.08778918, 2.94668175, 2.7702817 , 2.67643003, 2.61136678,
        2.53973465, 2.52449665, 2.50420345, 2.65070486, 2.65485127,
        2.6282719 , 2.80698555],

       [0.82749923, 0.97999447, 1.68655945, 2.1577152 , 2.29660643,
        2.7766564 , 2.9532252 , 2.87819831, 3.22360214, 3.36352136,
        3.26004713, 3.56231463, 3.62753204, 3.45379442, 3.68628759,
        3.71212929, 3.53929815, 3.77502242, 3.83618311, 3.72555794,
        3.99065978, 4.03378312, 3.9215523 , 4.17881701, 4.23856412,
        4.14500438, 4.39608411, 4.47199161, 4.39689574, 4.65658712,
        4.75688882, 4.67460713, 4.91652284, 4.99781698, 4.95948989,
        5.25307891, 5.33073883, 5.16596476, 5.18408102, 5.01542188,
        4.69907054, 4.60875852, 4.46560426, 4.29026363, 4.22829504,
        4.03934567, 3.86172841, 3.79479019, 3.67152308, 3.5816066 ,
        3.61021599, 3.53877814, 3.46347408, 3.77230796, 3.70442868,
        3.60614958, 3.96010856]])
    # fmt: on
    assert_almost_equal(dp, expected)
