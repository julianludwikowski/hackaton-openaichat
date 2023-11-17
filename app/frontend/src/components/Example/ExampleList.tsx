import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "Jak odblokować bankowość internetową?",
        value: "Jak odblokować bankowość internetową?"
    },
    { text: "Jak złożyć wniosek o pożyczkę gotówkową?", value: "Jak złożyć wniosek o pożyczkę gotówkową?" },
    { text: "Ile kosztuje Konto Jakże Osobiste?", value: "Ile kosztuje Konto Jakże Osobiste?" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
