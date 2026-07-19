
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_NAMES = ['1D_CNN', 'LSTM', 'CNN_LSTM', 'TCN', 'Transformer', 'Multimodal_Transformer']

plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24
})


def clean_column_name(col):
    col = str(col).strip()
    col = col.replace('Âµ', 'µ')
    col = col.replace('Âº', '°')
    col = col.replace('Â·', '·')
    col = col.replace('�', '')
    col = col.replace('\n', ' ')
    col = col.replace('\t', ' ')
    col = ' '.join(col.split())
    return col


def load_dataset(input_file, filter_active=True, active_rpm_min=500, active_esc_min=1100):
    df = pd.read_csv(input_file)
    df.columns = [clean_column_name(c) for c in df.columns]

    if 'class_label' not in df.columns:
        raise ValueError("The dataset must contain a 'class_label' column.")

    if 'source_file' not in df.columns:
        print('Warning: source_file column not found. Creating one pseudo group.')
        df['source_file'] = 'single_file'

    for col in df.columns:
        if col not in ['class_label', 'source_file']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['class_label']).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        med = df[col].median()
        if pd.isna(med):
            med = 0
        df[col] = df[col].fillna(med)

    if filter_active:
        before = len(df)
        rpm_col = 'Motor Electrical Speed (RPM)'
        esc_col = 'ESC signal (µs)'
        if rpm_col in df.columns:
            df = df[df[rpm_col] > active_rpm_min]
        elif esc_col in df.columns:
            df = df[df[esc_col] >= active_esc_min]
        df = df.reset_index(drop=True)
        print(f'Idle rows removed: {before - len(df)}')

    return df


def get_feature_cols(df):
    exclude = ['class_label', 'source_file']
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def create_windows(df, feature_cols, window_size=50, stride=50, label_col='class_label', group_col='source_file'):
    X_list, y_list, group_list = [], [], []
    for group_name, gdf in df.groupby(group_col, sort=False):
        gdf = gdf.reset_index(drop=True)
        if len(gdf) < window_size:
            print(f'Skipped {group_name}: only {len(gdf)} rows after filtering.')
            continue
        Xg = gdf[feature_cols].values.astype(np.float32)
        yg = gdf[label_col].astype(str).values
        for start in range(0, len(gdf) - window_size + 1, stride):
            end = start + window_size
            labels, counts = np.unique(yg[start:end], return_counts=True)
            y_win = labels[np.argmax(counts)]
            X_list.append(Xg[start:end])
            y_list.append(y_win)
            group_list.append(group_name)
    if len(X_list) == 0:
        raise ValueError('No windows were created. Reduce WINDOW_SIZE or collect more data.')
    return np.array(X_list), np.array(y_list), np.array(group_list)


def scale_by_train(X_train, X_val, X_test):
    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    scaler.fit(X_train.reshape(-1, n_features))
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    return X_train, X_val, X_test, scaler


def make_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]


def build_1d_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='1D_CNN')


def build_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.30)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='LSTM')


def build_cnn_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(96)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='CNN_LSTM')


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.20):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)


def build_tcn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu')(inputs)
    for dilation in [1, 2, 4, 8]:
        x = tcn_block(x, 64, 3, dilation)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='TCN')


def transformer_encoder(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.20):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)


def build_transformer(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128)(inputs)
    for _ in range(3):
        x = transformer_encoder(x, num_heads=4, key_dim=32, ff_dim=256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='Transformer')


def get_feature_groups(feature_cols):
    groups = {'control_speed': [], 'mechanical': [], 'electrical': [], 'vibration_acc': [], 'efficiency': [], 'load_cell': [], 'physics': []}
    for idx, col in enumerate(feature_cols):
        c = col.lower()
        if 'esc' in c or 'rpm' in c or 'angular speed' in c:
            groups['control_speed'].append(idx)
        elif 'load cell' in c:
            groups['load_cell'].append(idx)
        elif 'acc' in c or 'vibration' in c:
            groups['vibration_acc'].append(idx)
        elif 'voltage' in c or 'current' in c or 'electrical power' in c:
            groups['electrical'].append(idx)
        elif 'efficiency' in c or 'watt' in c:
            groups['efficiency'].append(idx)
        elif 'ratio' in c or 'error' in c:
            groups['physics'].append(idx)
        elif 'thrust' in c or 'torque' in c or 'mechanical power' in c:
            groups['mechanical'].append(idx)
    used = set(); clean_groups = {}
    for name, idxs in groups.items():
        uniq = []
        for i in idxs:
            if i not in used:
                uniq.append(i); used.add(i)
        if uniq:
            clean_groups[name] = uniq
    remaining = sorted(list(set(range(len(feature_cols))) - used))
    if remaining:
        clean_groups['other'] = remaining
    return clean_groups


def make_multimodal_inputs(X, feature_groups):
    return [X[:, :, idxs] for idxs in feature_groups.values()]


def branch_encoder(inp, branch_name):
    x = layers.Dense(64)(inp)
    x = transformer_encoder(x, num_heads=2, key_dim=16, ff_dim=128)
    return layers.GlobalAveragePooling1D(name=f'{branch_name}_gap')(x)


def build_multimodal_transformer(input_shape, num_classes, feature_cols):
    feature_groups = get_feature_groups(feature_cols)
    timesteps = input_shape[0]
    inputs, branches = [], []
    for group_name, idxs in feature_groups.items():
        inp = layers.Input(shape=(timesteps, len(idxs)), name=f'{group_name}_input')
        enc = branch_encoder(inp, group_name)
        inputs.append(inp); branches.append(enc)
    x = layers.Concatenate()(branches)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name='Multimodal_Transformer'), feature_groups


def build_model_by_name(model_name, input_shape, num_classes, feature_cols):
    if model_name == '1D_CNN': return build_1d_cnn(input_shape, num_classes), None
    if model_name == 'LSTM': return build_lstm(input_shape, num_classes), None
    if model_name == 'CNN_LSTM': return build_cnn_lstm(input_shape, num_classes), None
    if model_name == 'TCN': return build_tcn(input_shape, num_classes), None
    if model_name == 'Transformer': return build_transformer(input_shape, num_classes), None
    if model_name == 'Multimodal_Transformer': return build_multimodal_transformer(input_shape, num_classes, feature_cols)
    raise ValueError(f'Unknown model name: {model_name}')


def save_confusion_matrix(y_true, y_pred, class_names, out_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(class_names, fontweight='bold')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18, fontweight='bold')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()


def save_metrics_bar(metrics, out_path, title='Performance Metrics'):
    names = ['Accuracy', 'Precision', 'Recall', 'F1Score']
    values = [metrics.get(n, 0) for n in names]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold')
    plt.tight_layout(); plt.savefig(out_path, format='pdf', bbox_inches='tight'); plt.close()


def save_training_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(history.history.get('accuracy', []), label='Train Accuracy', linewidth=3)
    ax.plot(history.history.get('val_accuracy', []), label='Validation Accuracy', linewidth=3)
    ax.set_xlabel('Epoch', fontweight='bold'); ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Training Curve', fontweight='bold'); ax.legend(); ax.grid(True, linewidth=0.8)
    plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold')
    plt.tight_layout(); plt.savefig(out_path, format='pdf', bbox_inches='tight'); plt.close()


# def compute_metrics(y_true, y_pred):
#     return {
#         'Accuracy': accuracy_score(y_true, y_pred),
#         'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
#         'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
#         'F1Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
#         'Precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
#         'Recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
#         'F1Score_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
#     }

def compute_metrics(y_true, y_pred, all_labels=None):
    if all_labels is None:
        all_labels = np.unique(np.concatenate([y_true, y_pred]))

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "F1Score": f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0),
        "Precision_weighted": precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
        "F1Score_weighted": f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0),
    }


def train_one_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, feature_cols, result_dir, epochs=80, batch_size=16, learning_rate=1e-3, shuffle_labels=False, fold_name=None):
    tf.keras.backend.clear_session()
    input_shape = X_train.shape[1:]
    num_classes = len(label_encoder.classes_)
    model, feature_groups = build_model_by_name(model_name, input_shape, num_classes, feature_cols)
    y_train_used = y_train.copy()
    if shuffle_labels:
        np.random.shuffle(y_train_used)
    if feature_groups is not None:
        X_train_input = make_multimodal_inputs(X_train, feature_groups)
        X_val_input = make_multimodal_inputs(X_val, feature_groups)
        X_test_input = make_multimodal_inputs(X_test, feature_groups)
    else:
        X_train_input, X_val_input, X_test_input = X_train, X_val, X_test
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_input, y_train_used, validation_data=(X_val_input, y_val), epochs=epochs, batch_size=batch_size, callbacks=make_callbacks(), verbose=1)
    y_prob = model.predict(X_test_input, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    #metrics = compute_metrics(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, all_labels=all_labels)
    result_dir = Path(result_dir); result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(result_dir / 'metrics.csv', index=False)
    pd.DataFrame({'true_label': label_encoder.inverse_transform(y_test), 'predicted_label': label_encoder.inverse_transform(y_pred)}).to_csv(result_dir / 'predictions.csv', index=False)
    #report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0)

    all_labels = np.arange(len(label_encoder.classes_))

    report = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )

    with open(result_dir / 'classification_report.txt', 'w', encoding='utf-8') as f: f.write(report)
    save_confusion_matrix(y_test, y_pred, label_encoder.classes_, result_dir / 'confusion_matrix.pdf')
    save_metrics_bar(metrics, result_dir / 'metrics_bar.pdf')
    save_training_curve(history, result_dir / 'training_curve.pdf')
    try:
        model.save(result_dir / f'{model_name}.keras')
    except Exception as e:
        print('Model saving failed:', e)
    print('\n' + '='*70)
    print(f'Finished: {model_name}')
    if fold_name: print('Fold:', fold_name)
    print(pd.DataFrame([metrics])); print('='*70)
    return metrics


def prepare_windows(input_file, window_size=50, stride=50, filter_active=True, active_rpm_min=500, active_esc_min=1100):
    df = load_dataset(input_file, filter_active=filter_active, active_rpm_min=active_rpm_min, active_esc_min=active_esc_min)
    feature_cols = get_feature_cols(df)
    print('Dataset shape after cleaning/filtering:', df.shape)
    print('Number of numerical features:', len(feature_cols))
    print('Class counts after filtering:')
    print(df['class_label'].value_counts())
    X, y_text, groups = create_windows(df, feature_cols, window_size=window_size, stride=stride)
    print('\nCreated non-overlapping windows')
    print('Total windows:', len(y_text))
    print('Window class counts:')
    print(pd.Series(y_text).value_counts())
    print('Source files:', len(np.unique(groups)))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    return df, X, y, y_text, groups, label_encoder, feature_cols
