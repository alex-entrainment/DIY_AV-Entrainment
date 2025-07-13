import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function LogoTester() {
  const [svgContent, setSvgContent] = useState('');

  useEffect(() => {
    fetch('/dist/assets/OSC Logo No Background.svg')
      .then(res => res.text())
      .then(setSvgContent)
      .catch(err => console.error('Failed to load SVG', err));
  }, []);

  return (
    <div className="p-4 text-center">
      <h1 className="text-2xl font-bold mb-4">Logo Animation Tester</h1>
      {svgContent ? (
        <motion.div
          className="inline-block"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1 }}
          dangerouslySetInnerHTML={{ __html: svgContent }}
        />
      ) : (
        <p>Loading SVG...</p>
      )}
    </div>
  );
}
