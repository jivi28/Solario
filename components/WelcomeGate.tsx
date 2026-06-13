"use client"

import { useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import BavariaSuitabilityMap from "@/components/BavariaSuitabilityMap"

export default function WelcomeGate() {
  const [entered, setEntered] = useState(false)

  return (
    <>
      <BavariaSuitabilityMap />
      <AnimatePresence>
        {!entered && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
            className="fixed inset-0 z-[1000] flex flex-col items-center justify-center bg-black"
          >
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="font-[family-name:var(--font-display)] text-5xl font-bold tracking-tight text-white sm:text-7xl"
            >
              Welcome to SOLARIS
            </motion.h1>

            <motion.button
              type="button"
              onClick={() => setEntered(true)}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.25, ease: "easeOut" }}
              className="mt-10 rounded-full border border-yellow-400/40 bg-yellow-400 px-10 py-3 text-base font-bold text-black transition-colors hover:bg-yellow-300"
            >
              Continue
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
