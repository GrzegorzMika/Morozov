//
// Copyright 2019 Miral Shah <miralshah2211@gmail.com>
//
// Use, modification and distribution are subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/gil/image_processing/threshold.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/algorithm.hpp>
#include <boost/gil/gray.hpp>
#include <boost/core/lightweight_test.hpp>


namespace gil = boost::gil;

int height = 4;
int width = 4;

gil::gray8_image_t original_gray(width, height), threshold_gray(width, height),
expected_gray(width, height);

gil::rgb8_image_t original_rgb(width, height), threshold_rgb(width, height),
expected_rgb(width, height);

void fill_original_gray()
{
    //filling original view's upper half part with gray pixels of value 50
    //filling original view's lower half part with gray pixels of value 150
    gil::fill_pixels(gil::subimage_view(gil::view(original_gray), 0, 0, original_gray.width(),
        original_gray.height() / 2), gil::gray8_pixel_t(50));
    gil::fill_pixels(gil::subimage_view(gil::view(original_gray), 0, original_gray.height() / 2,
        original_gray.width(), original_gray.height() / 2), gil::gray8_pixel_t(150));
}

void fill_original_rgb()
{
    //filling original_rgb view's upper half part with rgb pixels of value 50, 85, 135
    //filling original_rgb view's lower half part with rgb pixels of value 150, 205, 106
    gil::fill_pixels(gil::subimage_view(gil::view(original_rgb), 0, 0, original_rgb.width(),
        original_rgb.height() / 2), gil::rgb8_pixel_t(50, 85, 135));
    gil::fill_pixels(gil::subimage_view(gil::view(original_rgb), 0, original_rgb.height() / 2,
        original_rgb.width(), original_rgb.height() / 2), gil::rgb8_pixel_t(150, 205, 106));
}

void threshold_gray_to_gray()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected view's upper half part with gray pixels of value 50
    //filling expected view's lower half part with gray pixels of value 100
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, 0, original_gray.width(),
        original_gray.height() / 2), gil::gray8_pixel_t(50));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, original_gray.height() / 2,
        original_gray.width(), original_gray.height() / 2), gil::gray8_pixel_t(100));

    gil::threshold_truncate(gil::view(original_gray), gil::view(threshold_gray), 100);

    //comparing threshold view generated by the function with the expected view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_gray), gil::view(expected_gray)));
}

void threshold_inverse_gray_to_gray()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected view's upper half part with gray pixels of value 100
    //filling expected view's lower half part with gray pixels of value 150
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, 0, original_gray.width(),
        original_gray.height() / 2), gil::gray8_pixel_t(100));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, original_gray.height() / 2,
        original_gray.width(), original_gray.height() / 2), gil::gray8_pixel_t(150));

    gil::threshold_truncate
    (
        gil::view(original_gray),
        gil::view(threshold_gray),
        100,
        gil::threshold_truncate_mode::threshold,
        gil::threshold_direction::inverse
    );

    //comparing threshold view generated by the function with the expected view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_gray), gil::view(expected_gray)));
}

void zero_gray_to_gray()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected view's upper half part with gray pixels of value 0
    //filling expected view's lower half part with gray pixels of value 150
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, 0, original_gray.width(),
        original_gray.height() / 2), gil::gray8_pixel_t(0));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, original_gray.height() / 2,
        original_gray.width(), original_gray.height() / 2), gil::gray8_pixel_t(150));

    gil::threshold_truncate
    (
        gil::view(original_gray),
        gil::view(threshold_gray),
        100,
        gil::threshold_truncate_mode::zero,
        gil::threshold_direction::regular
    );

    //comparing threshold view generated by the function with the expected view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_gray), gil::view(expected_gray)));
}

void zero_inverse_gray_to_gray()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected view's upper half part with gray pixels of value 50
    //filling expected view's lower half part with gray pixels of value 0
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, 0, original_gray.width(),
        original_gray.height() / 2), gil::gray8_pixel_t(50));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_gray), 0, original_gray.height() / 2,
        original_gray.width(), original_gray.height() / 2), gil::gray8_pixel_t(0));

    gil::threshold_truncate
    (
        gil::view(original_gray),
        gil::view(threshold_gray),
        100,
        gil::threshold_truncate_mode::zero,
        gil::threshold_direction::inverse
    );

    //comparing threshold view generated by the function with the expected view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_gray), gil::view(expected_gray)));
}

void threshold_rgb_to_rgb()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected_rgb view's upper half part with rgb pixels of value 50
    //filling expected_rgb view's lower half part with rgb pixels of value 100
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, 0, original_rgb.width(),
        original_rgb.height() / 2), gil::rgb8_pixel_t(50, 85, 100));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, original_rgb.height() / 2,
        original_rgb.width(), original_rgb.height() / 2), gil::rgb8_pixel_t(100, 100, 100));

    gil::threshold_truncate(gil::view(original_rgb), gil::view(threshold_rgb), 100);

    //comparing threshold_rgb view generated by the function with the expected_rgb view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_rgb), gil::view(expected_rgb)));
}

void threshold_inverse_rgb_to_rgb()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected_rgb view's upper half part with rgb pixels of value 103, 59, 246
    //filling expected_rgb view's lower half part with rgb pixels of value 150
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, 0, original_rgb.width(),
        original_rgb.height() / 2), gil::rgb8_pixel_t(100, 100, 135));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, original_rgb.height() / 2,
        original_rgb.width(), original_rgb.height() / 2), gil::rgb8_pixel_t(150, 205, 106));

    gil::threshold_truncate
    (
        gil::view(original_rgb),
        gil::view(threshold_rgb),
        100,
        gil::threshold_truncate_mode::threshold,
        gil::threshold_direction::inverse
    );

    //comparing threshold_rgb view generated by the function with the expected_rgb view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_rgb), gil::view(expected_rgb)));
}

void zero_rgb_to_rgb()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected_rgb view's upper half part with rgb pixels of value 0
    //filling expected_rgb view's lower half part with rgb pixels of value 150
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, 0, original_rgb.width(),
        original_rgb.height() / 2), gil::rgb8_pixel_t(0, 0, 135));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, original_rgb.height() / 2,
        original_rgb.width(), original_rgb.height() / 2), gil::rgb8_pixel_t(150, 205, 106));

    gil::threshold_truncate
    (
        gil::view(original_rgb),
        gil::view(threshold_rgb),
        100,
        gil::threshold_truncate_mode::zero,
        gil::threshold_direction::regular
    );

    //comparing threshold_rgb view generated by the function with the expected_rgb view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_rgb), gil::view(expected_rgb)));
}

void zero_inverse_rgb_to_rgb()
{
    //expected view after thresholding of the original view with threshold value of 100
    //filling expected_rgb view's upper half part with rgb pixels of value 50
    //filling expected_rgb view's lower half part with rgb pixels of value 0
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, 0, original_rgb.width(),
        original_rgb.height() / 2), gil::rgb8_pixel_t(50, 85, 0));
    gil::fill_pixels(gil::subimage_view(gil::view(expected_rgb), 0, original_rgb.height() / 2,
        original_rgb.width(), original_rgb.height() / 2), gil::rgb8_pixel_t(0, 0, 0));

    gil::threshold_truncate
    (
        gil::view(original_rgb),
        gil::view(threshold_rgb),
        100,
        gil::threshold_truncate_mode::zero,
        gil::threshold_direction::inverse
    );

    //comparing threshold_rgb view generated by the function with the expected_rgb view
    BOOST_TEST(gil::equal_pixels(gil::view(threshold_rgb), gil::view(expected_rgb)));
}


int main()
{
    fill_original_gray();
    fill_original_rgb();

    threshold_gray_to_gray();
    threshold_inverse_gray_to_gray();
    zero_gray_to_gray();
    zero_inverse_gray_to_gray();

    threshold_rgb_to_rgb();
    threshold_inverse_rgb_to_rgb();
    zero_rgb_to_rgb();
    zero_inverse_rgb_to_rgb();

    return boost::report_errors();
}