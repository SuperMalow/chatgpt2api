type ImageComposerNativeKeyboardEvent = {
  isComposing?: boolean;
  keyCode?: number;
  which?: number;
};

type ImageComposerSubmitKeyEvent = {
  key: string;
  shiftKey: boolean;
  nativeEvent?: ImageComposerNativeKeyboardEvent;
};

export function shouldSubmitImageComposerPrompt(
  event: ImageComposerSubmitKeyEvent,
) {
  const nativeEvent = event.nativeEvent;
  const isImeComposing =
    nativeEvent?.isComposing ||
    nativeEvent?.keyCode === 229 ||
    nativeEvent?.which === 229;

  return event.key === "Enter" && !event.shiftKey && !isImeComposing;
}
