import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

// Everything stays inside the Shorts safe zone: YouTube's UI covers roughly the
// bottom 20% and a strip on the right, so text sits between 8% and 72% height.
const FONT = '"Arial Black", "Segoe UI Black", Impact, sans-serif';
const outline = '0 0 3px #000, 0 0 3px #000, 0 4px 14px rgba(0,0,0,0.85)';

const SceneClip = ({ src, durationInFrames }) => {
  const frame = useCurrentFrame();
  // Snap-in on each cut plus a slow drift so AI clips never feel static
  const snap = interpolate(frame, [0, 8], [1.12, 1.04], { extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic) });
  const drift = interpolate(frame, [8, durationInFrames], [0, 0.04], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ transform: `scale(${snap + drift})` }}>
      <OffthreadVideo src={staticFile(src)} muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
    </AbsoluteFill>
  );
};

const Label = ({ label, sublabel, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pop = spring({ frame: frame - 4, fps, config: { damping: 12, stiffness: 180 } });
  return (
    <AbsoluteFill style={{ justifyContent: 'flex-start', alignItems: 'center', top: '63%' }}>
      <div style={{ transform: `scale(${pop})`, textAlign: 'center', maxWidth: '84%' }}>
        <div style={{ fontFamily: FONT, fontSize: 74, color: accent, textShadow: outline, lineHeight: 1.05 }}>{label}</div>
        {sublabel ? (
          <div style={{ fontFamily: FONT, fontSize: 46, color: 'white', textShadow: outline, marginTop: 10 }}>{sublabel}</div>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};

const Header = ({ header, headerStyle }) => {
  if (!header) return null;
  if (headerStyle === 'pov') {
    return (
      <div style={{ position: 'absolute', top: '8%', left: 0, right: 0, background: 'white', padding: '22px 48px', textAlign: 'center' }}>
        <div style={{ fontFamily: 'Segoe UI, Arial, sans-serif', fontWeight: 700, fontSize: 50, color: '#111' }}>{header}</div>
      </div>
    );
  }
  return (
    <div style={{ position: 'absolute', top: '8%', width: '100%', textAlign: 'center' }}>
      <div style={{ fontFamily: FONT, fontSize: 64, color: 'white', textShadow: outline, letterSpacing: 2 }}>{header}</div>
    </div>
  );
};

// Word-by-word captions from Whisper timings; the active word is highlighted
const Captions = ({ words, accent }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const idx = words.findIndex((w) => t >= w.start && t < w.end + 0.15);
  if (idx < 0) return null;
  const groupStart = idx - (idx % 3);
  const group = words.slice(groupStart, groupStart + 3);
  return (
    // Upper third: AI characters are framed centrally, so mid-screen captions cover their faces
    <AbsoluteFill style={{ top: '16%', alignItems: 'center' }}>
      <div style={{ fontFamily: FONT, fontSize: 60, textAlign: 'center', maxWidth: '86%', textShadow: outline }}>
        {group.map((w, i) => (
          <span key={i} style={{ color: groupStart + i === idx ? accent : 'white', marginRight: 18 }}>
            {w.text.toUpperCase()}
          </span>
        ))}
      </div>
    </AbsoluteFill>
  );
};

export const Short = ({ scenes, header, headerStyle, accent, words, music, voice, disclaimer }) => {
  const { durationInFrames, fps } = useVideoConfig();
  let from = 0;
  return (
    <AbsoluteFill style={{ backgroundColor: 'black' }}>
      {scenes.map((s) => {
        const seq = (
          <Sequence key={s.id} from={from} durationInFrames={s.durationInFrames}>
            <SceneClip src={s.src} durationInFrames={s.durationInFrames} />
            <Label label={s.label} sublabel={s.sublabel} accent={accent} />
          </Sequence>
        );
        from += s.durationInFrames;
        return seq;
      })}
      <Header header={header} headerStyle={headerStyle} />
      {words && words.length ? <Captions words={words} accent={accent} /> : null}
      {disclaimer ? (
        <Sequence from={durationInFrames - 2 * fps}>
          <AbsoluteFill style={{ top: '76%', alignItems: 'center' }}>
            <div style={{ fontFamily: 'Segoe UI, Arial, sans-serif', fontSize: 30, color: 'white', textShadow: outline, opacity: 0.9 }}>
              {disclaimer}
            </div>
          </AbsoluteFill>
        </Sequence>
      ) : null}
      {music ? (
        <Audio
          src={staticFile(music)}
          volume={(f) =>
            interpolate(f, [0, 10, durationInFrames - 20, durationInFrames], [0, voice ? 0.18 : 0.8, voice ? 0.18 : 0.8, 0], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            })
          }
        />
      ) : null}
      {voice ? <Audio src={staticFile(voice)} /> : null}
    </AbsoluteFill>
  );
};

export const calculateShortMetadata = ({ props }) => ({
  durationInFrames: props.scenes.reduce((n, s) => n + s.durationInFrames, 0),
});
