+ 将henet_stage2.py 放置在 EDVR/basicsr/models/archs 路径下

+ edvr_arch.py 修改处：

  - line 8：

    ```python
    from hrnet_2stage import feature_extraction
    ```

  - line 327 - 329:

    ```python
    # self.feature_extraction = make_layer(
    #     ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
    self.feature_extraction = feature_extraction(pretrained=True)
    ```

  - line 380 - 381:

    ```python
    feat_l1 = x.view(-1, c, h, w)
    feat_l1 = self.feature_extraction(feat_l1)
    ```

    