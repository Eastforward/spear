# AudioSet 室内动物声源覆盖表（严格版 v1）

官方 revision：`d417d32bf59c711abb5910fd2f76a0eb44697991`；可视声源：32 类。
声学事件（如 bark、meow、neigh）挂到动物实体上，不重复生成网格。所有条目当前均为 research candidate。

| 声源 | AudioSet 类 | 室内范围 | 运动家族 | 动作 | 当前状态 |
|---|---|---|---|---|---|
| dog | Dog (`/m/0bt9lr`) | apartment_common | quadruped_canid | Idle / Walking | strict_canary_validated |
| cat | Cat (`/m/01yrx`) | apartment_common | quadruped_felid | Idle / Walking | strict_canary_validated |
| horse | Horse (`/m/03k3r`) | specialized_indoor_only | quadruped_equid | Idle / Walking | strict_canary_in_progress |
| donkey | Donkey, ass (`/m/0ffhf`) | specialized_indoor_only | quadruped_equid | Idle / Walking | native_motion_guide_available |
| cattle | Cattle, bovinae (`/m/01xq0k1`) | specialized_indoor_only | quadruped_bovid | Idle / Walking | native_motion_guide_available |
| yak | Yak (`/m/01hhp3`) | specialized_indoor_only | quadruped_bovid | Idle / Walking | strict_profile_and_guide_required |
| pig | Pig (`/m/068zj`) | specialized_indoor_only | quadruped_suid | Idle / Walking | strict_profile_and_guide_required |
| goat | Goat (`/m/03fwl`) | specialized_indoor_only | quadruped_small_ungulate | Idle / Walking | strict_profile_and_guide_required |
| sheep | Sheep (`/m/07bgp`) | specialized_indoor_only | quadruped_small_ungulate | Idle / Walking | strict_profile_and_guide_required |
| fowl_generic | Fowl (`/m/025rv6n`) | specialized_indoor_only | bird_terrestrial | Idle / Walking / Hop | bird_adapter_required |
| chicken_rooster | Chicken, rooster (`/m/09b5t`) | specialized_indoor_only | bird_terrestrial | Idle / Walking / Hop | bird_adapter_required |
| turkey | Turkey (`/m/01rd7k`) | specialized_indoor_only | bird_terrestrial | Idle / Walking / Hop | bird_adapter_required |
| duck | Duck (`/m/09ddx`) | specialized_indoor_only | bird_terrestrial | Idle / Walking / Hop | bird_adapter_required |
| goose | Goose (`/m/0dbvp`) | specialized_indoor_only | bird_terrestrial | Idle / Walking / Hop | bird_adapter_required |
| roaring_big_cat | Roaring cats (lions, tigers) (`/m/0cdnk`) | specialized_indoor_only | quadruped_felid | Idle / Walking | strict_profile_and_guide_required |
| bird_generic | Bird (`/m/015p6`) | apartment_conditional | bird_perching_flight | PerchedIdle / Hop / Flying | bird_adapter_required |
| pigeon_dove | Pigeon, dove (`/m/0h0rv`) | apartment_conditional | bird_perching_flight | PerchedIdle / Hop / Flying | bird_adapter_required |
| crow | Crow (`/m/04s8yn`) | apartment_conditional | bird_perching_flight | PerchedIdle / Hop / Flying | bird_adapter_required |
| owl | Owl (`/m/09d5_`) | specialized_indoor_only | bird_perching_flight | PerchedIdle / Hop / Flying | bird_adapter_required |
| gull | Gull, seagull (`/m/01dwxx`) | specialized_indoor_only | bird_perching_flight | PerchedIdle / Hop / Flying | bird_adapter_required |
| wild_canid | Canidae, dogs, wolves (`/m/01z5f`) | specialized_indoor_only | quadruped_canid | Idle / Walking | native_motion_guide_available |
| rodent_generic | Rodents, rats, mice (`/m/06hps`) | apartment_conditional | quadruped_rodent | Idle / Scurry | strict_profile_and_guide_required |
| mouse | Mouse (`/m/04rmv`) | apartment_common | quadruped_rodent | Idle / Scurry | strict_profile_and_guide_required |
| chipmunk | Chipmunk (`/m/02021`) | apartment_conditional | quadruped_rodent | Idle / Scurry | strict_profile_and_guide_required |
| insect_generic | Insect (`/m/03vt0`) | apartment_common | insect_crawl_flight | Idle / Crawl / Flying | insect_adapter_required |
| cricket | Cricket (`/m/09xqv`) | apartment_common | insect_crawl_flight | Idle / Crawl / Flying | insect_adapter_required |
| mosquito | Mosquito (`/m/09f96`) | apartment_common | insect_crawl_flight | Idle / Crawl / Flying | insect_adapter_required |
| housefly | Fly, housefly (`/m/0h2mp`) | apartment_common | insect_crawl_flight | Idle / Crawl / Flying | insect_adapter_required |
| bee_wasp | Bee, wasp, etc. (`/m/01h3n`) | apartment_conditional | insect_crawl_flight | Idle / Crawl / Flying | insect_adapter_required |
| frog | Frog (`/m/09ld4`) | apartment_conditional | amphibian_hop | Idle / Hop | amphibian_adapter_required |
| snake | Snake (`/m/078jl`) | apartment_conditional | serpent_slither | Idle / Slither | serpent_adapter_required |
| whale | Whale vocalization (`/m/032n05`) | specialized_indoor_only | aquatic_cetacean | IdleSwim / Swimming | aquatic_adapter_required |

普通 Apartment 只能自动采样 `apartment_common`；`apartment_conditional` 必须由场景语义显式允许；`specialized_indoor_only` 禁止进入普通 Apartment。
鸟类、昆虫、蛇、青蛙和鲸分别使用自己的运动适配器，不能套四足 Walking。
