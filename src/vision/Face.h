#pragma once

// A detected face in image pixel coordinates
struct Face {
    float cx;      // box center
    float cy;
    int width;
    int height;
    float score;
};
