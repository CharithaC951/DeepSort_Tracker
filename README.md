# DeepSort_Tracker
Deep SORT (“Simple Online and Realtime Tracking with a Deep association metric”) is used for multi-object tracking in videos. Its job: keep a consistent ID on each object (e.g., each person) across frames—even with occlusions or crossing paths.

#Inputs: per-frame detections (boxes + scores) from a detector (YOLO, Faster R-CNN, etc.).

#Core idea:

Predict object motion with a Kalman filter (where should each track be in the next frame?).

Compute assignment cost between current detections and existing tracks using motion distance (Mahalanobis) plus an appearance embedding (a CNN re-ID feature).

Use the Hungarian algorithm to match detections to tracks with lowest total cost.

Why “Deep”: it adds a learned appearance descriptor so a person can be re-associated after partial/long occlusion, not just by proximity.

Outputs: a list of tracks each frame → (track_id, bbox, score) with stable IDs over time.

Use cases: people tracking, shopper analytics, sports players, vehicles, occupancy, handoff to re-ID, action recognition, etc.

Tradeoffs:

Very reliable ID continuity vs. basic SORT.

Needs a decent detector and an appearance model; a bit heavier than SORT.

Modern alternatives: ByteTrack, OC-SORT, BoT-SORT—often higher MOTA with strong detectors, but Deep SORT remains a solid, simple baseline that’s easy to integrate. 
