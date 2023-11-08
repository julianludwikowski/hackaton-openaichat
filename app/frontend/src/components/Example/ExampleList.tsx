import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "Co to jest Karta Wirtualna?",
        value: "Co to jest Karta Wirtualna?"
    },
    { text: "Do jakiej sumy mogę wziąć pożyczkę internetową?", value: "Do jakiej sumy mogę wziąć pożyczkę internetową?" },
    { text: "Jak zgłosić reklamację?", value: "Jak zgłosić reklamację?" }
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
