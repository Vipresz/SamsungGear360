// Helper functions for UV validation and coordinate conversion

// Helper function to validate UV coordinates in full texture space
bool validUV(vec2 u) {
    return all(greaterThanEqual(u, vec2(0.0))) && all(lessThanEqual(u, vec2(1.0)));
}

// Helper function to validate UV coordinates for a specific lens half
bool validLensUV(vec2 u, bool lens1) {
    if (!validUV(u)) return false;
    if (uIsHorizontal) {
        return lens1 ? (u.x <= 0.5) : (u.x >= 0.5);
    } else {
        return lens1 ? (u.y <= 0.5) : (u.y >= 0.5);
    }
}
