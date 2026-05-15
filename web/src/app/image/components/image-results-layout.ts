const IMAGE_TURN_ASPECT_CLASSES: Record<string, string> = {
  "1:1": "aspect-square",
  "16:9": "aspect-video",
  "9:16": "aspect-[9/16]",
  "4:3": "aspect-[4/3]",
  "3:4": "aspect-[3/4]",
};

const IMAGE_TURN_MOBILE_SQUARE_ASPECT_CLASSES: Record<string, string> = {
  "1:1": "aspect-square sm:aspect-square",
  "16:9": "aspect-square sm:aspect-video",
  "9:16": "aspect-square sm:aspect-[9/16]",
  "4:3": "aspect-square sm:aspect-[4/3]",
  "3:4": "aspect-square sm:aspect-[3/4]",
};

const IMAGE_TURN_ASPECT_RATIOS: Record<string, string> = {
  "1:1": "1 / 1",
  "16:9": "16 / 9",
  "9:16": "9 / 16",
  "4:3": "4 / 3",
  "3:4": "3 / 4",
};

export function getImageTurnAspectClass(
  size: string,
  { mobileSquare = false }: { mobileSquare?: boolean } = {},
) {
  const aspectClasses = mobileSquare
    ? IMAGE_TURN_MOBILE_SQUARE_ASPECT_CLASSES
    : IMAGE_TURN_ASPECT_CLASSES;

  return aspectClasses[size] || aspectClasses["1:1"];
}

export function getImageTurnAspectRatio(size: string) {
  return IMAGE_TURN_ASPECT_RATIOS[size] || IMAGE_TURN_ASPECT_RATIOS["1:1"];
}
