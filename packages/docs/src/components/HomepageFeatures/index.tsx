import React from "react";
import clsx from "clsx";
import styles from "./styles.module.css";
import Translate from "@docusaurus/Translate";

const FeatureList = [
    {
        title: <Translate>Easy to Use</Translate>,
        Svg: require("@site/static/img/easy_use.svg").default,
        description: (
            <Translate>
                Built for Node.js and easily accessible via npm. Support
                Typescript.
            </Translate>
        ),
    },
    {
        title: <Translate>Open Source Language Model</Translate>,
        Svg: require("@site/static/img/llama.svg").default,
        description: (
            <Translate>
                Load large language model LLaMA, RWKV and LLaMA's derived models.
            </Translate>
        ),
    },
    {
        title: <Translate>Cross platforms</Translate>,
        Svg: require("@site/static/img/cross-platform.svg").default,
        description: (
            <Translate>
                Supports Windows, Linux, and macOS. Allow full accelerations on
                CPU inference (SIMD powered by llama.cpp/llm-rs/rwkv.cpp).
            </Translate>
        ),
    },
];

function Feature({ Svg, title, description }) {
    return (
        <div className={clsx("col col--4")}>
            <div className="text--center">
                <Svg className={styles.featureSvg} role="img" />
            </div>
            <div className="text--center padding-horiz--md">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
    );
}

export default function HomepageFeatures() {
    return (
        <section className={styles.features}>
            <div className="container">
                <div className="row">
                    {FeatureList.map((props, idx) => (
                        <Feature key={idx} {...props} />
                    ))}
                </div>
            </div>
        </section>
    );
}
