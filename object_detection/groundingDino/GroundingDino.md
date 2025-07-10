### What Each Threshold Controls:

| Threshold | What It Measures | Lower Value = | Higher Value = |
| --- | --- | --- | --- |
| **TEXT_THRESHOLD** | How similar the detected object must be to your text query | More lenient matching | Stricter text matching |
| **BOX_THRESHOLD** | Quality of bounding box proposals | More boxes detected | Fewer, higher-quality boxes |
| **CONFIDENCE_THRESHOLD** | Final detection confidence | More results kept | Fewer, more confident results |
## Text Similarity "Distance" Explanation
The `TEXT_THRESHOLD` (0.25 in your config) controls the **semantic distance** between:
- Your text query (e.g., "person with weapon")
- What the model sees in the image
