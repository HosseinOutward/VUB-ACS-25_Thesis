- make a saved run of the model training (raw codec) for future use
- have an automated pipeline (possibly with mlflow) for visualizing codec performance over fl rounds
- have a debug mode for skipping fl training and instead load the saved run models. should be a very seperate code to the main spine
- have a protocol with R rounds, but the train target should be the latest FEW recons grads, instead of just the last one.
- make a vector quantizer (transformer so that it sees the entire current gradient vector as opposed to norrow window or progressive window). this also includes a prior model for it.
- for the VQ, also try few recons grad as target, instead of a just the last one.
- for testing the ideas, ditch fl acc checks, fl traing (load instead), skip rANS. we should streamline testing costs
- try out the idea of fine tuning on a true sample, or even training on it, as opposed to training only on past recons (with the target being a stale round recons). idea from the innovation gradient part of LGC-DDL paper and how it trains the model.
- instead of training 5 times and selecting the top llwz, train once, if the performance is much worse, train again, if still much worse, train one last time and use the one that was the least bad. if any train gives nan, then retrain anyways. https://claude.ai/chat/25b9a045-c3d9-406d-b496-9ffa82357315
    - train a model with last recon as target, send the model, get an encoded smaple back, fine tune the head of the model, send only the head (head of encoder, the rest retrain).
    - train a model with all recons in side info (y1, yn), and the target is randomly one of them. then do the above. this way the model has the last recon as side info too.
    - do the first, but have a placeholder side info for the last recon which is the target at first, then during sample fine tuning, have a head with access to side info.  

https://claude.ai/chat/33feaeba-0d5d-438c-97bf-062a0106d27f