Problem:
checking the bearing before it damages the whole machinery with the help of noise input
what we do:
A damaged bearing vibrates differently. We built a neural network that reads 0.17 seconds of vibration and says whether the bearing is healthy, which part is damaged (ball, inner ring or outer ring) and how big the damage is.
how testing is been done:
We split each recording by time: the first 70% for learning, the next 15% for validation and the last 15% for testing. Then we cut it into small pieces.
what we learnt:
A model can score perfectly in familiar conditions and still make mistakes when they change. Using frequencies helped only slightly, so this problem isn't solved yet.