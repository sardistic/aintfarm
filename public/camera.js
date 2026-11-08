export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;

export function clampCamera(camera, width, height) {
  const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, camera.zoom));
  return {
    zoom,
    x: Math.min(0, Math.max(width * (1 - zoom), camera.x)),
    y: Math.min(0, Math.max(height * (1 - zoom), camera.y))
  };
}

export function zoomCamera(camera, zoom, anchorX, anchorY, width, height) {
  const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
  const ratio = nextZoom / camera.zoom;
  return clampCamera({
    zoom: nextZoom,
    x: anchorX - (anchorX - camera.x) * ratio,
    y: anchorY - (anchorY - camera.y) * ratio
  }, width, height);
}

export function panCamera(camera, dx, dy, width, height) {
  return clampCamera({ zoom: camera.zoom, x: camera.x + dx, y: camera.y + dy }, width, height);
}
