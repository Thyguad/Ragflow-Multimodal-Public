import { IImageRenderPlan } from '@/interfaces/database/chat';
import { Modal } from 'antd';
import classNames from 'classnames';
import { useState } from 'react';

import styles from './index.less';

const MessageImageRenderPlan = ({ plan }: { plan?: IImageRenderPlan }) => {
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);

  if (!plan?.show_images || !plan.items?.length || plan.mode === 'none') {
    return null;
  }

  const items = plan.items;
  const previewItem =
    previewIndex === null
      ? null
      : items[Math.max(0, Math.min(previewIndex, items.length - 1))];

  const handlePreviewOpen = (index: number) => () => {
    setPreviewIndex(index);
  };

  const handlePreviewClose = () => {
    setPreviewIndex(null);
  };

  if (plan.mode === 'single') {
    const item = items[0];
    if (!item) {
      return null;
    }
    return (
      <section className={classNames(styles.wrapper, styles.singleWrapper)}>
        <img
          src={item.image_url}
          alt={item.display_caption || item.caption || ''}
          className={classNames(styles.singleImage, styles.previewableImage)}
          loading="lazy"
          onClick={handlePreviewOpen(0)}
        />
        <div className={styles.caption}>
          {item.display_caption || item.caption}
        </div>
        <Modal
          open={Boolean(previewItem)}
          footer={null}
          onCancel={handlePreviewClose}
          width="min(96vw, 1200px)"
          centered
          className={styles.previewModal}
        >
          {previewItem && (
            <figure className={styles.previewFigure}>
              <img
                src={previewItem.image_url}
                alt={previewItem.display_caption || previewItem.caption || ''}
                className={styles.previewImage}
              />
              <figcaption className={styles.previewCaption}>
                {previewItem.display_caption || previewItem.caption}
              </figcaption>
            </figure>
          )}
        </Modal>
      </section>
    );
  }

  if (plan.mode === 'compare') {
    const compareItems = items.slice(0, 2);
    if (compareItems.length !== 2) {
      return null;
    }
    return (
      <section className={classNames(styles.wrapper, styles.compareWrapper)}>
        {compareItems.map((item, index) => (
          <figure
            key={`${item.doc_name}-${item.figure_key}`}
            className={styles.compareFigure}
          >
            <img
              src={item.image_url}
              alt={item.display_caption || item.caption || ''}
              className={classNames(
                styles.compareImage,
                styles.previewableImage,
              )}
              loading="lazy"
              onClick={handlePreviewOpen(index)}
            />
            <figcaption className={styles.caption}>
              {item.display_caption || item.caption}
            </figcaption>
          </figure>
        ))}
        <Modal
          open={Boolean(previewItem)}
          footer={null}
          onCancel={handlePreviewClose}
          width="min(96vw, 1200px)"
          centered
          className={styles.previewModal}
        >
          {previewItem && (
            <figure className={styles.previewFigure}>
              <img
                src={previewItem.image_url}
                alt={previewItem.display_caption || previewItem.caption || ''}
                className={styles.previewImage}
              />
              <figcaption className={styles.previewCaption}>
                {previewItem.display_caption || previewItem.caption}
              </figcaption>
            </figure>
          )}
        </Modal>
      </section>
    );
  }

  return (
    <section className={classNames(styles.wrapper, styles.galleryWrapper)}>
      {items.slice(0, 3).map((item, index) => (
        <figure
          key={`${item.doc_name}-${item.figure_key}`}
          className={styles.galleryFigure}
        >
          <img
            src={item.image_url}
            alt={item.display_caption || item.caption || ''}
            className={classNames(styles.galleryImage, styles.previewableImage)}
            loading="lazy"
            onClick={handlePreviewOpen(index)}
          />
          <figcaption className={styles.caption}>
            {item.display_caption || item.caption}
          </figcaption>
        </figure>
      ))}
      <Modal
        open={Boolean(previewItem)}
        footer={null}
        onCancel={handlePreviewClose}
        width="min(96vw, 1200px)"
        centered
        className={styles.previewModal}
      >
        {previewItem && (
          <figure className={styles.previewFigure}>
            <img
              src={previewItem.image_url}
              alt={previewItem.display_caption || previewItem.caption || ''}
              className={styles.previewImage}
            />
            <figcaption className={styles.previewCaption}>
              {previewItem.display_caption || previewItem.caption}
            </figcaption>
          </figure>
        )}
      </Modal>
    </section>
  );
};

export default MessageImageRenderPlan;
