import { AbsoluteFill, Composition, interpolate, useCurrentFrame } from 'remotion';
import { calculateShortMetadata, Short } from './Short';

const HelloTitle = ({ title }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 30], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill style={{ backgroundColor: '#101820', justifyContent: 'center', alignItems: 'center' }}>
      <h1 style={{ color: 'white', fontFamily: 'sans-serif', fontSize: 100, opacity }}>{title}</h1>
    </AbsoluteFill>
  );
};

export const Root = () => (
  <>
    <Composition
      id="HelloTitle"
      component={HelloTitle}
      durationInFrames={60}
      fps={30}
      width={1280}
      height={720}
      defaultProps={{ title: 'AI Video Editor' }}
    />
    {/* Vertical Short; props come from pipeline/assemble.py */}
    <Composition
      id="Short"
      component={Short}
      fps={30}
      width={1080}
      height={1920}
      durationInFrames={150}
      calculateMetadata={calculateShortMetadata}
      defaultProps={{ scenes: [], header: '', headerStyle: 'top', accent: '#FFD84D', words: [], music: null, voice: null, disclaimer: '' }}
    />
  </>
);
