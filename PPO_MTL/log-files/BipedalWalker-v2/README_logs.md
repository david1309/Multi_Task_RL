* PPO_loss_stl: single domain learning  agents (yeah, it says STL but its actually SDL) testing which of policy ratio clipping was the best - KL (TRPO) vs Clip (PPO) vs both (TRPO + PPO)

* PPO_loss_mtl: multi domain learning  agents (yeah, it says MTL but its actually MDL) testing which of policy ratio clipping was the best - KL (TRPO) vs Clip (PPO) vs both (TRPO + PPO)

*stl_tx: once the best loss strategy had been chosen, we tested the performance of *several SDL agents* in each domain (wind 0, 1 and 2) with different network depths (3, 5 and 7 hidden layers)

* sdt_tx_casc: same stl_tx experiments but instead of having a iso-depth architectures (64-64-64 units per hidden layer) we have architectures as encoder (125-50-25) and encoder-decoder (125-50-125)

*mtl_t012: This experiments are with BATCH SIZE = 5, which makes them extremely unstableo --> tested the performance of *a single MDL agent* trainned in parallel across all domains (wind 0, 1 and 2) with different network depths (3, 5 and 7 hidden layers)

*mdl_t012: This experiments are with BATCH SIZE = 20 making them more stalbe --> same as mtl_t012 

*mdl_t012_cascade: This experiments are with BATCH SIZE = 20 making them more stalbe -->  same as mtl_t012 but with the above described encoder and encoder-decoder architectures

**all of this experiments were done before correcting the bug  I had in the code when applying PPO's clipping strategy (please see disseration to understand the bug I had)**

* PPO_loss_mdl and PPO_loss_sdl: same as PPO_loss_mtl and PPO_loss_mtl but correcting the bug of the PPO clipping.


