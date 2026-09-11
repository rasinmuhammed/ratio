"use client";

import { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// The number of nodes in the citation network visualization
const NODE_COUNT = 80;
const CONNECTION_THRESHOLD = 0.28;

function CitationNetwork() {
  const meshRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const { size, mouse } = useThree();

  // Generate stable node positions
  const { positions, nodeColors } = useMemo(() => {
    const pos = new Float32Array(NODE_COUNT * 3);
    const cols = new Float32Array(NODE_COUNT * 3);

    for (let i = 0; i < NODE_COUNT; i++) {
      // Spread across a sphere-like volume
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.5 + Math.random() * 2.5;

      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6; // Flatten vertically
      pos[i * 3 + 2] = r * Math.cos(phi) * 0.5;

      // Node color: mostly fog, some gold (supreme court nodes), some sapphire
      const rand = Math.random();
      if (rand < 0.08) {
        // Gold — Supreme Court
        cols[i * 3] = 0.78; cols[i * 3 + 1] = 0.66; cols[i * 3 + 2] = 0.29;
      } else if (rand < 0.2) {
        // Sapphire — High Court
        cols[i * 3] = 0.28; cols[i * 3 + 1] = 0.43; cols[i * 3 + 2] = 0.65;
      } else {
        // Fog — default
        cols[i * 3] = 0.25; cols[i * 3 + 1] = 0.25; cols[i * 3 + 2] = 0.28;
      }
    }
    return { positions: pos, nodeColors: cols };
  }, []);

  // Build connection line geometry
  const { linePositions, lineColors } = useMemo(() => {
    const linePos: number[] = [];
    const lineCols: number[] = [];

    for (let i = 0; i < NODE_COUNT; i++) {
      for (let j = i + 1; j < NODE_COUNT; j++) {
        const dx = positions[i * 3] - positions[j * 3];
        const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < CONNECTION_THRESHOLD * 10) {
          linePos.push(
            positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
            positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]
          );
          const alpha = 1 - dist / (CONNECTION_THRESHOLD * 10);
          const col = 0.12 + alpha * 0.1;
          lineCols.push(col, col, col + 0.04, col, col, col + 0.04);
        }
      }
    }

    return {
      linePositions: new Float32Array(linePos),
      lineColors: new Float32Array(lineCols),
    };
  }, [positions]);

  useFrame(({ clock }) => {
    if (!meshRef.current || !linesRef.current) return;
    const t = clock.getElapsedTime();

    // Slow, peaceful rotation
    const group = meshRef.current.parent;
    if (group) {
      group.rotation.y = t * 0.04 + mouse.x * 0.15;
      group.rotation.x = mouse.y * -0.08;
    }
  });

  return (
    <group>
      {/* Connection lines */}
      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[lineColors, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial vertexColors transparent opacity={0.5} />
      </lineSegments>

      {/* Nodes */}
      <points ref={meshRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[nodeColors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          size={0.045}
          sizeAttenuation
          transparent
          opacity={0.9}
        />
      </points>
    </group>
  );
}

export function CitationWebGL() {
  return (
    <Canvas
      camera={{ position: [0, 0, 7], fov: 55 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: "transparent" }}
      dpr={[1, 1.5]}
    >
      <ambientLight intensity={0.2} />
      <CitationNetwork />
    </Canvas>
  );
}
